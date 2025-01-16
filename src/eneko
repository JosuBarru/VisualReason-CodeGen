import datetime
import math
import os
import pathlib
from functools import partial
import warnings
import traceback


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
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['CODEX_QUANTIZED'] = '1'
os.environ['LOAD_MODELS'] = '1'
os.environ['DATASET'] = 'okvqa'
os.environ['EXEC_MODE'] = 'cache'
os.environ['COGNITION_MODEL'] = 'config_mistral'

script_dir = os.path.abspath('/gaueko0/users/eamor002/viper')
sys.path.append(script_dir)

from configs import config
from src.utils import seed_everything, print_all_data, repair_csv
import datasets
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


def run_program(parameters, queues_in_, input_type_, retrying=True):
    from src.image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, process_guesses
    from src.video_segment import VideoSegment

    global queue_results
    syntax_error = 0

    code, sample_id, image, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, process_guesses, bool_to_yesno, distance, best_image_match):\n' \
                  f'    # Answer is:'
    code = code_header + code

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        syntax_error = 1
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            exec(compile(code, 'Codex', 'exec'), globals())
            syntax_error = 0
        except Exception as e2:
            syntax_error = 1
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code, syntax_error

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)
    process_guesses_partial = partial(process_guesses, queues=queues)
    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, process_guesses_partial, bool_to_yesno, distance, best_image_match)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        syntax_error = 1
        if retrying:
            return None, code, syntax_error
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code, syntax_error


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


# New function
def save_results(all_data,dataset):
    results_dir = pathlib.Path(config['results_dir'])
    results_dir = results_dir / config.dataset.split
    results_dir.mkdir(parents=True, exist_ok=True)
    if config.save_codex:
        if not config.save_new_results:
            filename = 'codex_results.csv'
        else:
            existing_files = list(results_dir.glob('codex_results_*.csv'))
            if len(existing_files) == 0:
                filename = 'codex_results_0.csv'
            else:
                filename = 'codex_results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        print('Saving results to', filename)
        all_sample_ids, all_queries, all_codes = all_data
        if config.dataset.dataset_name == 'RefCOCO':
            all_versions = [config.dataset.version for _ in range(dataset.__len__())]
            all_splits = [str(f'{config.dataset.split} by {config.dataset.split_by}') for _ in range(dataset.__len__())]
            data = [all_sample_ids, all_queries, all_splits, all_versions, all_codes]
            columns = ['sample_id','query','split', 'version', 'generated_code']
        else:
            all_splits = [config.dataset.split for _ in range(dataset.__len__())]
            data = [all_sample_ids, all_queries,all_splits, all_codes]
            columns = ['sample_id','query','split', 'generated_code']
        df = pd.DataFrame(data).T
        df.columns = columns
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

    elif config.save:
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            existing_files = list(results_dir.glob('results_*.csv'))
            if len(existing_files) == 0:
                filename = 'results_0.csv'
            else:
                filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        print('Saving results to', filename)
        
        if config.dataset.dataset_name == 'RefCOCO':
            print_all_data(all_data)
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, all_IoUs, acc_vector, score_result, is_syntax_error = all_data
            all_versions = [config.dataset.version for _ in range(dataset.__len__())]
            all_splits = [str(f'{config.dataset.split} by {config.dataset.split_by}') for _ in range(dataset.__len__())]
            data = [all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers,all_codes, 
                    all_splits,all_versions,all_IoUs, acc_vector, is_syntax_error]
            # data = {'sample_id':all_sample_ids,'query': all_queries , 'Answer': all_results, 'image_path': all_img_paths, 'truth_answers':all_truth_answers, 'code': all_codes, 'split':all_splits, 'version':all_versions, 'IoU': all_IoUs, 'accuracy': acc_vector}
            columns = ['sample_id','query', 'Answer', 'image_path', 'truth_answers', 'code', 'split','version', 'IoU', 'accuracy', 'is_syntax_error']
            global_score_line = {'sample_id':[0.0],'query': [0.0] , 'Answer': [0.0], 'image_path':[0.0], 'truth_answers':[0.0], 'code': [0.0], 'split':[0.0], 'version':[0.0], 'IoU': score_result[0], 'accuracy': score_result[1], 'is_syntax_error': [0.0]}
            df = pd.DataFrame(data).T
            df.columns = columns
            df['Answer'] = df['Answer'].apply(str) # some answers can be numbers
            last_line = pd.Series(global_score_line)
            df = pd.concat([df, last_line], ignore_index=True)
            print(df.columns)
            repair_csv(df,results_dir=results_dir, filename=filename)
        else:
            all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers, all_codes, acc_vector, score_result, is_syntax_error = all_data
            all_splits = [config.dataset.split for _ in range(dataset.__len__())]
            # data = [all_sample_ids, all_queries, all_results, all_img_paths, all_truth_answers,all_codes, 
            #         all_splits, all_accuracies]
            data = {'sample_id': all_sample_ids ,'query': all_queries, 'Answer': all_results, 'image_path': all_img_paths, 'truth_answers': all_truth_answers, 'code': all_codes, 'split': all_splits, 'accuracy': acc_vector, 'is_syntax_error': is_syntax_error}
            columns =  ['sample_id','query', 'Answer', 'image_path', 'truth_answers', 'code', 'split', 'accuracy', 'is_syntax_error']
            #global_score_line = {'sample_id':'-','query': '-' , 'Answer': '-', 'image_path':'-', 'truth_answers':'-', 'code': '-', 'split':'-', 'accuracy': score_result}
            global_score_line = {'sample_id':[0.0],'query': [0.0] , 'Answer': [0.0], 'image_path':[0.0], 'truth_answers':[0.0], 'code': [0.0], 'split':[0.0], 'accuracy': score_result, 'is_syntax_error': [0.0]}        
            df = pd.DataFrame(data)
            #df.columns = columns
            df['Answer'] = df['Answer'].apply(str) # some answers can be numbers
            last_line = pd.DataFrame(global_score_line)
            df = pd.concat([df, last_line], ignore_index=True)
            df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

def main():

    ''' To run this script: "CONFIG_NAMES=your_config_name python main_batch.py"
        Or
        Adding this code-lines at the beginning of the function:

        "   os.environ['CONFIG_NAMES'] = your_config_name
            os.environ['CUDA_VISIBLE_DEVICES'] = '0' # For example
            script_dir = os.path.abspath('path/to/your_project')    "
    '''

    mp.set_start_method('spawn')

    from src.vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    # Added codeLLama Quantized  
    if config.codex.model == 'codellama':
        model_name_codex = 'codellama'
    elif config.codex.model == 'codellama_Q':
        model_name_codex  = 'codellama_Q'
    else:
        model_name_codex = 'codex'
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)

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

    # Lists to keep data for .csv
    all_results = []
    all_answers = []
    all_codes = []
    all_sample_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    all_images = []
    all_IoUs = []
    max_memory_usage = 0.0
    syntax_error_counter = 0
    is_syntax_error =[]
    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)
            
            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # Combine all queries and get Codex predictions for them (code generation)
                if not config.use_cached_codex:
                    codes = codex(prompt=batch['query'], base_prompt=base_prompt, input_type=input_type,
                                  extra_context=batch['extra_context'])
                    memory_usage = torch.cuda.memory_allocated() / (1024*1024)
                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache

                # Run the code
                if config.execute_code:
                    if not config.multiprocessing:
                        # Otherwise, we would create a new model for every process
                        results = []
                        for c, sample_id, img, possible_answers, query in \
                                zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                            result= run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                            result_ = (result[0], result[1])
                            syntax_error_counter += result[2]
                            is_syntax_error.append(result[2])
                            result = result_
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
                #all_images.append(batch['image']) 
                
                # Metrics` part
                if i % config.log_every == 0:
                    try:
                        if config.dataset.dataset_name=='RefCOCO':
                            accuracy, _ , _ = dataset.accuracy(prediction=all_results, ground_truth=all_answers)
                        else:
                            accuracy, _ = dataset.accuracy(prediction=all_results, ground_truth=all_answers)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')
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
            all_data = [all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes,score_vector, accuracy, is_syntax_error]
        else:
            all_data = [all_sample_ids, all_queries, all_results, all_img_paths, all_answers, all_codes, all_IoUs, score_vector, accuracy, is_syntax_error]
    print(f'syntax_error counter:{syntax_error_counter}/{dataset.__len__()}')
    save_results(all_data, dataset)
    #     if config.wandb:
    #         wandb.log({'accuracy': accuracy})
    #         wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})
    
    finish_all_consumers()


if __name__ == '__main__':
    main()
