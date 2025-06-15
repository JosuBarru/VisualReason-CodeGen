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
import re

import sys

#Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)
logging.getLogger("maskrcnn_benchmark").setLevel(logging.WARNING)
logging.getLogger("vllm").disabled = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CODEX_QUANTIZED'] = '1'
# os.environ['LOAD_MODELS'] = '1'
# os.environ['DATASET'] = 'gqa'
# os.environ['EXEC_MODE'] = 'cache'
# os.environ['COGNITION_MODEL'] = 'config_gemma'
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper')
sys.path.append(script_dir)
os.chdir(script_dir)


from configs import config
from utils import seed_everything
#import datasets

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def run_program(parameters, queues_in_, input_type_, retrying=False):
    from src.image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from src.video_segment import VideoSegment
    logger.debug("Running")
    global queue_results

    code, sample_id, image, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match):\n'

    if 'def execute_command' in code:
        # Remove everything up to and including the line with 'def execute_command...'
        code = re.sub(r'^.*?def execute_command[^\n]*\n', '', code, flags=re.DOTALL)
        # Optionally trim after the triple backticks
        code = code.split('```')[0]
    logger.debug(f"Code: {code}")
    code = code_header + str(code)




    # Define a timeout handler that raises a TimeoutError.
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out after 5 minutes")
    # Set the alarm signal for 300 seconds (5 minutes).
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    try:
        try:
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e:
            return f"Error Codigo: {e}", code

        queues = [queues_in_, queue_results]

        image_patch_partial = partial(ImagePatch, queues=queues)
        video_segment_partial = partial(VideoSegment, queues=queues)
        llm_query_partial = partial(llm_query, queues=queues)

        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match)
    except TimeoutError as te:
        return f"Timeout: {te}", code
    except Exception as e:
        return f"Error Ejecucion: {e}", code
    finally:
        signal.alarm(0)  # Cancel the alarm

    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    return result, code



def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]

def save_results(all_data,dataset):
    results_dir = pathlib.Path(config['results_dir'])
    results_dir = results_dir / config.dataset.split
    results_dir.mkdir(parents=True, exist_ok=True)
    if config.save_codex:
        if not config.save_new_results:
            filename = 'codex_results.csv'
        else:
            filename = config.codex.model + '___' + datetime.datetime.now().strftime("%m-%d_%H-%M") +'.csv'

        logger.info(f'Saving results to {filename}')
        all_sample_ids, all_queries, all_codes = all_data
        if config.dataset.dataset_name == 'RefCOCO':
            data = [all_sample_ids, all_queries, all_codes]
            columns = ['sample_id','query', 'generated_code']
        else:
            data = [all_sample_ids, all_queries, all_codes]
            columns = ['sample_id','query', 'generated_code']
        df = pd.DataFrame(data).T
        df.columns = columns
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

    elif config.save:
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            filename = "eval_" + config.cached_codex_path.split("/")[-1].split(".")[0].split("_")[-1] + "___" + datetime.datetime.now().strftime("%m-%d_%H-%M") +'.csv'

        logger.info(f'Saving results to {filename}')    

        if config.dataset.dataset_name == 'RefCOCO':
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, all_IoUs, acc_vector, score_result = all_data
            data = [all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers,all_codes,all_IoUs, acc_vector]
            columns = ['sample_id','query', 'Answer', 'image_path', 'truth_answers', 'code', 'IoU', 'accuracy']
            global_score_line = {'sample_id':'-','query': '-' , 'Answer': '-', 'image_path':'-', 'truth_answers':'-', 'code': '-', 'IoU': score_result[0], 'accuracy': score_result[1]}
        else:
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, acc_vector, score_result = all_data
            data = [all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, acc_vector]
            columns =  ['sample_id','query', 'Answer', 'image_path', 'truth_answers', 'code', 'accuracy']
            global_score_line = {'sample_id':'-','query': '-' , 'Answer': '-', 'image_path':'-', 'truth_answers':'-', 'code': '-', 'accuracy': score_result}
        
        df = pd.DataFrame(data).T
        df.columns = columns
        df['Answer'] = df['Answer'].apply(str) # some answers can be numbers
        last_line = pd.DataFrame([global_score_line])
        df = pd.concat([df, last_line], ignore_index=True)
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

def main():

    logger.info("Starting main")

    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from my_datasets import get_dataset

    logger.info("Models successfully loaded")

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)
    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    # Added codeLLama Quantized  
    # if config.codex.model == 'codellama':
    #     model_name_codex = 'codellama'
    # elif config.codex.model == 'codellama_Q':
    #     model_name_codex  = 'codellama_Q'
    # elif config.codex.model == 'llama_31-8bq':
    #     model_name_codex = 'llama31_q'
    # else:
    #     model_name_codex = 'codex'

    model_name_codex = config.codex.model

    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)

    logger.info("Dataset loaded")

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()

    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        # codes_all = [r.split('# Answer is:')[1] for r in results['code']]
        codes_all = [r for r in results['generated_code']]
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    all_results = []
    all_answers = []
    all_codes = []
    all_sample_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    all_IoUs = []

    #num_instances = 0

    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches, ascii=True, ncols=100, mininterval=10):

                #num_instances += batch_size 
                # if num_instances % 100 < batch_size: 
                #     tqdm.write(f"Processing batch {i}/{n_batches}")

                logger.debug(f"input: {batch['query']}")

                if not config.use_cached_codex:
                    codes = codex(prompt=batch['query'], base_prompt=base_prompt, input_type=input_type,
                                  extra_context=batch['extra_context'])

                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache

                # Run the code
                if config.execute_code:
                    if not config.multiprocessing:
                        # Otherwise, we would create a new model for every process
                        results = []
                        for c, sample_id, img, possible_answers, query in \
                                zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                            result = run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                            results.append(result)
                    else:
                        results = list(pool.imap(partial(
                            run_program, queues_in_=queues_in, input_type_=input_type),
                            zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])))
                else:
                    results = [(None, c) for c in codes]
                    warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                  "'execute_code' to False by default, because executing code generated by a language "
                                  "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                  "it.")

                all_results += [r[0] for r in results]
                all_codes += [r[1] for r in results]
                all_sample_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['query_type']
                all_queries += batch['query']
                all_img_paths += [dataset.get_sample_path(idx) for idx in batch['index']]

                # if i % config.log_every == 0:
                #     try:
                #         accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                #         console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                #     except Exception as e:
                #         console.print(f'Error computing accuracy: {e}')

        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        if config.dataset.dataset_name!='RefCOCO':
            accuracy, score_vector= dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        else:
            accuracy, all_IoUs, score_vector = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save_codex:
        all_data = [all_sample_ids, all_queries, all_codes]
    elif config.save:
        if config.dataset.dataset_name!='RefCOCO':
            all_data = [all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes,score_vector , accuracy]
        else:
            all_data = [all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes,all_IoUs,score_vector, accuracy]
    save_results(all_data, dataset)
    #     if config.wandb:
    #         wandb.log({'accuracy': accuracy})
    #         wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})

    finish_all_consumers()


if __name__ == '__main__':
    main()
