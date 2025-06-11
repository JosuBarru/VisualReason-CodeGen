import datetime
import math
import os
import pathlib
from functools import partial
import warnings
import traceback
import signal

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("maskrcnn_benchmark").setLevel(logging.WARNING)
logging.getLogger("vllm").disabled = True

script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper')
sys.path.append(script_dir)
os.chdir(script_dir)

from configs import config
from utils import seed_everything

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
mp.set_sharing_strategy('file_system')
queue_results = None
cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)

def my_collate(batch):
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def run_program(parameters, queues_in_, input_type_, retrying=False):
    from src.image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from src.video_segment import VideoSegment

    code, sample_id, image, possible_answers, query = parameters
    code_header = (
        f'def execute_command_{sample_id}({input_type_}, possible_answers, query, '
        'ImagePatch, VideoSegment, llm_query, bool_to_yesno, distance, best_image_match):\n'
    )

    if 'def execute_command' in code:
        code = code.split('def execute_command')[1]
        code = code.split('```')[0]
        code = code_header + str(code)

    # No signal.alarm here; timeout enforced externally
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        queues = [queues_in_, queue_results]
        image_patch_partial = partial(ImagePatch, queues=queues)
        video_segment_partial = partial(VideoSegment, queues=queues)
        llm_query_partial = partial(llm_query, queues=queues)

        result = globals()[f'execute_command_{sample_id}'](
            image, possible_answers, query,
            image_patch_partial, video_segment_partial,
            llm_query_partial, bool_to_yesno, distance, best_image_match
        )
    except Exception as e:
        return f"Error Ejecucion: {e}", code
    finally:
        if f'execute_command_{sample_id}' in globals():
            del globals()[f'execute_command_{sample_id}']

    return result, code


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def save_results(all_data, dataset):
    results_dir = pathlib.Path(config['results_dir']) / config.dataset.split
    results_dir.mkdir(parents=True, exist_ok=True)

    if config.save_codex:
        filename = config.codex.model + '___' + datetime.datetime.now().strftime("%m-%d_%H-%M") + '.csv' if config.save_new_results else 'codex_results.csv'
        all_sample_ids, all_queries, all_codes = all_data
        df = pd.DataFrame(list(zip(all_sample_ids, all_queries, all_codes)), columns=['sample_id','query','generated_code'])
        df.to_csv(results_dir / filename, index=False, encoding='utf-8')

    elif config.save:
        filename = "eval_" + config.cached_codex_path.split("/")[-1].split(".")[0].split("_")[-1] + "___" + datetime.datetime.now().strftime("%m-%d_%H-%M") + '.csv' if config.save_new_results else 'results.csv'
        if dataset.dataset_name == 'RefCOCO':
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, all_IoUs, acc_vector, score_result = all_data
            columns = ['sample_id','query','Answer','image_path','truth_answers','code','IoU','accuracy']
            global_line = {'sample_id':'-','query':'-','Answer':'-','image_path':'-','truth_answers':'-','code':'-','IoU':score_result[0],'accuracy':score_result[1]}
        else:
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, acc_vector, score_result = all_data
            columns = ['sample_id','query','Answer','image_path','truth_answers','code','accuracy']
            global_line = {'sample_id':'-','query':'-','Answer':'-','image_path':'-','truth_answers':'-','code':'-','accuracy':score_result}

        df = pd.DataFrame(list(zip(all_sample_ids,all_queries,all_results,all_img_paths,all_truth_answers,all_codes,acc_vector)), columns=columns[:-1])
        df['accuracy'] = acc_vector if dataset.dataset_name!='RefCOCO' else None
        df = pd.concat([df, pd.DataFrame([global_line])], ignore_index=True)
        df.to_csv(results_dir / filename, index=False, encoding='utf-8')


def main():
    logger.info("Starting main")
    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from my_datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None] * batch_size

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=my_collate)
    input_type = dataset.input_type

    all_results, all_answers, all_codes = [], [], []
    all_sample_ids, all_queries, all_img_paths = [], [], []
    all_possible_answers, all_query_types, all_IoUs = [], [], []

    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), ascii=True, ncols=100, mininterval=10):
        logger.debug(f"input: {batch['query']}")
        codes = codex(prompt=batch['query'], base_prompt=base_prompt, input_type=input_type, extra_context=batch['extra_context']) if not config.use_cached_codex else codes_all[i*batch_size:(i+1)*batch_size]

        if config.execute_code:
            params = list(zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']))
            results = []
            with ProcessPoolExecutor(max_workers=num_processes, initializer=worker_init, initargs=(queues_results,)) as executor:
                futures = [executor.submit(run_program, p, queues_in, input_type) for p in params]
                for fut in futures:
                    try:
                        results.append(fut.result(timeout=300))  # 5 minutes
                    except FuturesTimeout:
                        results.append(("Timeout: exceeded 5m", None))
        else:
            results = [(None, c) for c in codes]
            warnings.warn("Not executing code! set config.execute_code=True to enable.")

        all_results.extend(r[0] for r in results)
        all_codes.extend(r[1] for r in results)
        all_sample_ids.extend(batch['sample_id'])
        all_answers.extend(batch['answer'])
        all_possible_answers.extend(batch['possible_answers'])
        all_query_types.extend(batch['query_type'])
        all_queries.extend(batch['query'])
        all_img_paths.extend(dataset.get_sample_path(idx) for idx in batch['index'])

    try:
        if dataset.dataset_name != 'RefCOCO':
            accuracy, score_vector = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        else:
            accuracy, all_IoUs, score_vector = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        console.print(f'Error computing accuracy: {e}')

    if config.save_codex:
        save_results([all_sample_ids, all_queries, all_codes], dataset)
    elif config.save:
        if dataset.dataset_name!='RefCOCO':
            save_results([all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes, score_vector, accuracy], dataset)
        else:
            save_results([all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes, all_IoUs, score_vector, accuracy], dataset)

    finish_all_consumers()

if __name__ == '__main__':
    main()
