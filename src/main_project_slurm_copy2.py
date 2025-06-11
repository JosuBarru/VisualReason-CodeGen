import datetime
import math
import os
import pathlib
import traceback
import sys
import logging
from functools import partial
import warnings
import signal

import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from joblib import Memory
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("maskrcnn_benchmark").setLevel(logging.WARNING)
logging.getLogger("vllm").disabled = True

# Setup paths
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper')
sys.path.append(script_dir)
os.chdir(script_dir)

from configs import config
from utils import seed_everything

# Multiprocessing strategy for PyTorch
mp.set_sharing_strategy('file_system')

cache = Memory('cache/' if config.use_cache else None, verbose=0)
seed_everything()
console = Console(highlight=False)

def my_collate(batch):
    return {k: [d[k] for d in batch] for k in batch[0].keys()}

def wrapper(q, *args):
        try:
            result = target(*args)
            q.put(('success', result))
        except Exception as e:
            q.put(('error', (e, traceback.format_exc())))

def run_with_timeout(target, args=(), timeout=300):
    q = mp.Queue()
    p = mp.Process(target=wrapper, args=(q, *args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return ("Timeout: Function execution exceeded time limit.", None)
    if q.empty():
        return ("Error: no result returned before join()", None)
    status, value = q.get()
    if status == 'success':
        return value
    else:
        err, tb = value
        return (f"Error Ejecucion: {err}\n{tb}", None)

def actual_execution(parameters, queues_in_, input_type_):
    from src.image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from src.video_segment import VideoSegment

    code, sample_id, image, possible_answers, query = parameters
    header = (
        f"def execute_command_{sample_id}("
        f"{input_type_}, possible_answers, query, "
        "ImagePatch, VideoSegment, llm_query, bool_to_yesno, distance, best_image_match):\n"
    )

    if 'def execute_command' in code:
        code = code.split('def execute_command', 1)[1]
        code = code.split('```', 1)[0]
    code = header + str(code)

    queues = [queues_in_, globals().get('queue_results')]
    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    exec(compile(code, 'Codex', 'exec'), globals())
    func = globals()[f'execute_command_{sample_id}']
    result = func(image, possible_answers, query,
                  image_patch_partial, video_segment_partial,
                  llm_query_partial, bool_to_yesno, distance, best_image_match)

    del globals()[f'execute_command_{sample_id}']
    return result, code

def run_program(parameters, queues_in_, input_type_, retrying=False):
    return run_with_timeout(actual_execution, args=(parameters, queues_in_, input_type_), timeout=300)

def worker_init(queue_results_):
    global queue_results
    idx = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[idx]

def save_results(all_data, dataset):
    results_dir = pathlib.Path(config['results_dir'])
    results_dir = results_dir / config.dataset.split
    results_dir.mkdir(parents=True, exist_ok=True)
    dt = datetime.datetime.now().strftime("%m-%d_%H-%M")
    if config.save_codex:
        filename = config.codex.model + '___' + dt + '.csv' if config.save_new_results else 'codex_results.csv'
    else:
        filename = config.cached_codex_path.split("/")[-1].split("_")[-1] + '___' + dt + '.csv' if config.save_new_results else 'results.csv'

    logger.info(f"Saving results to {filename}")
    df = pd.DataFrame(all_data).T
    df.columns = ['sample_id','query','generated_code'] if config.save_codex else (['sample_id','query','Answer','image_path','truth_answers','code','accuracy'] if config.dataset.dataset_name!='RefCOCO' else ['sample_id','query','Answer','image_path','truth_answers','code','IoU','accuracy'])
    if not config.save_codex:
        df['Answer'] = df['Answer'].astype(str)
        if config.dataset.dataset_name == 'RefCOCO':
            df = pd.concat([df, pd.DataFrame([{**dict(zip(df.columns, ['-']*len(df.columns))), 'IoU': all_data[-1], 'accuracy': all_data[-2]}])], ignore_index=True)
    df.to_csv(results_dir / filename, index=False, encoding='utf-8')

def main():
    logger.info("Starting main")
    mp.set_start_method('spawn')
    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from my_datasets import get_dataset

    logger.info("Models successfully loaded")
    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    queue_results_main = manager.Queue() if config.multiprocessing else None
    queues_results = [manager.Queue() for _ in range(batch_size)] if config.multiprocessing else [None]*batch_size

    model_name_codex = config.codex.model
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)
    logger.info("Dataset loaded")

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()

    codes_all = pd.read_csv(config.cached_codex_path)['generated_code'].tolist() if config.use_cached_codex else None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=my_collate
    )
    input_type = dataset.input_type

    accumulators = {name: [] for name in ['results', 'answers', 'codes', 'sample_ids', 'queries', 'img_paths', 'possible_answers', 'query_types']}

    if config.dataset.dataset_name == 'RefCOCO':
        accumulators['IoUs'] = []

    pool = mp.Pool(
        processes=num_processes,
        initializer=worker_init,
        initargs=(queues_results,)
    ) if config.multiprocessing else None

    try:
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), ascii=True, ncols=100):
            if not config.use_cached_codex:
                codes = codex(prompt=batch['query'], base_prompt=base_prompt, input_type=input_type,
                              extra_context=batch.get('extra_context', None))
            else:
                codes = codes_all[i*batch_size:(i+1)*batch_size]

            if config.execute_code:
                if pool is None:
                    results = [run_program([c, sid, img, pa, q], queues_in, input_type) for c, sid, img, pa, q
                               in zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])]
                else:
                    results = pool.imap(
                        partial(run_program, queues_in_=queues_in, input_type_=input_type),
                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])
                    )
                    results = list(results)
            else:
                results = [(None, c) for c in codes]
                warnings.warn("Not executing code! Set 'execute_code' to True to run.")

            for (res, code), sid, ans, pa, qt, qp, ip in zip(results, batch['sample_id'], batch['answer'],
                                                             batch['possible_answers'], batch['query_type'],
                                                             batch['query'], [dataset.get_sample_path(idx) for idx in batch['index']]):
                accumulators['results'].append(res)
                accumulators['codes'].append(code)
                accumulators['sample_ids'].append(sid)
                accumulators['answers'].append(ans)
                accumulators['possible_answers'].append(pa)
                accumulators['query_types'].append(qt)
                accumulators['queries'].append(qp)
                accumulators['img_paths'].append(ip)

    except Exception as e:
        traceback.print_exc()
        console.print(f"Excepción general: {e}")
    finally:
        if pool:
            pool.close()
            pool.join()

    # Cálculo de métricas
    try:
        if config.dataset.dataset_name == 'RefCOCO':
            acc, IoUs, _ = dataset.accuracy(
                accumulators['results'], accumulators['answers'],
                accumulators['possible_answers'], accumulators['query_types']
            )
            accumulators['IoUs'] = IoUs
        else:
            acc, _ = dataset.accuracy(
                accumulators['results'], accumulators['answers'],
                accumulators['possible_answers'], accumulators['query_types']
            )
        console.print(f"Final accuracy: {acc}")
    except Exception as e:
        console.print(f"Error al computar accuracy: {e}")

    # Guardar
    if config.save_codex:
        all_data = [accumulators['sample_ids'], accumulators['queries'], accumulators['codes']]
    else:
        if config.dataset.dataset_name != 'RefCOCO':
            all_data = [
                accumulators['sample_ids'], accumulators['queries'], accumulators['results'],
                accumulators['img_paths'], accumulators['answers'],
                accumulators['codes'], accumulators['possible_answers'], acc
            ]
        else:
            all_data = [
                accumulators['sample_ids'], accumulators['queries'], accumulators['results'],
                accumulators['img_paths'], accumulators['answers'],
                accumulators['codes'], accumulators['IoUs'], accumulators['possible_answers'], acc
            ]
    save_results(all_data, dataset)
    finish_all_consumers()

if __name__ == "__main__":
    main()
