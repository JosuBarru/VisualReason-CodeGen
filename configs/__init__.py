import os
from omegaconf import OmegaConf
import sys
import yaml

import logging

logger = logging.getLogger(__name__)


dataset_name = os.getenv('DATASET', None) # Dataset sobre el que trabajamos
execution_mode = os.getenv('EXEC_MODE', None) # codex o cache  (generar codigos o ejecutar)
enable_models = bool(int(os.getenv('LOAD_MODELS', '0'))) # which models to load (depends on wether we want to generate or execute, SO RELATED WITH ABOVE)
cognition_models = os.getenv('COGNITION_MODEL', None) # Just for general knowledge datasets (okvqa)
codex_model = os.getenv('CODEX_MODEL', None)  # Name of the model to generate code
partition = os.getenv('PARTITION', None)
code = os.getenv('CODE', None)
num_inst_str = os.getenv('NUMINST')  # Defaults to None if the variable is not set
num_inst = int(num_inst_str) if num_inst_str not in (None, "") else None
batch_size = os.getenv('BATCHSIZE')
batch_size = int(batch_size) if batch_size is not None else None
temp_str = os.getenv('TEMP', None)
temp = int(temp_str) if temp_str is not None else None

checkpoint = os.getenv('CHECKPOINT', None)

mult = bool(int(os.getenv('MULT', '0')))



config_names = []


model_configs = {
    "codellama": "config_codellama",
    "codellama_base": "config_codellama_base",
    "llama31Q": "config_codex_llama3.1-8b",
    "llama31Q_Base": "config_codex_llama3.1-8b-base",
    "llama33Q": "config_codex_llama3.3-70b",
    "llama3Q": "config_codex_llama3-8b",
    "deepseek-qwen7b": "config_codex_deepseek-Qwen-7b",
    "deepseek-llama8b": "config_codex_deepseek-llama-8b",
    "deepseek-llama70b": "config_codex_deepseek-llama-70b",
    "qwen25":"config_codex_qwen2.5-7b",
    "qwen25_inst":"config_codex_qwen2.5-7b_inst",
    "mixtral87b":"config_codex_mixtral8-7b",
}



# codes_dir = {
#     "deepseek-llama8b" : 'results/gqa/codex_results/train/codex_results_deepSeekLlama8b-post.csv',
#     "llama31-8b-16b" : 'results/gqa/codex_results/train/codex_results_llama31-16bit.csv'
# }


configs = []

try:
    if dataset_name in ['refcoco','gqa', 'okvqa']:
        config_names.append(dataset_name + '/'+ 'general_config')
        if batch_size is not None:
            manual_batch = OmegaConf.create({
                        "dataset": {"batch_size": batch_size}
                    })
        if execution_mode is not None:
            if execution_mode == 'cache':
                if cognition_models is not None:
                    config_names.insert(0, cognition_models)
                if code is not None and partition in ['train', 'val', 'testdev']:
                    manual_config = OmegaConf.create({
                        "use_cached_codex": True, 
                        "cached_codex_path": os.path.join('results/gqa/codex_results/', partition, code)
                    })
                else:
                    raise UserWarning(f"Add input file and partition")

            elif execution_mode == 'codex':
                if codex_model in model_configs:
                    config_names.append(model_configs[codex_model])
                    if checkpoint is not None and checkpoint != '':
                        manual_adapter = OmegaConf.create({
                                "codex": {"adapter": os.path.join("/sorgin1/users/jbarrutia006/viper/", checkpoint)}
                            })
                    if temp is not None:
                        manual_temp = OmegaConf.create({
                                "codex": {"temperature": temp}
                            })
                    else:
                        raise UserWarning(f"Add temperature for codex model")
                else:
                    raise UserWarning(
                        f"Model '{codex_model}' is not recognized. Please check the configuration mapping."
                    )
                config_names.append(dataset_name + '/'+ 'save_codex')
                if num_inst is not None:
                    manual_long = OmegaConf.create({
                                "dataset": {"max_samples": num_inst}
                            })
            elif not execution_mode in [None, 'cache', 'codex']:
                raise NameError(f'Value from $EXEC_MODE variable is incorrect, obtained: {execution_mode} and must be: cache or codex')
            
            
            if partition in ['train', 'val', 'testdev']:
                config_names.append(dataset_name + '/' + partition) 
        config_names_=','.join(config_names)
        config_names = config_names_
    else: 
        raise UserWarning(f"There is not any dataset setted or obtained value ({dataset_name}) from '$DATASET' ENV $variable is INCORRECT")
except NameError as n:
    print(f'ERROR: {n}')
    exit()
except UserWarning as w:
    print(f'WARNING !!!!: {w}')



# if dataset_name is None:
#     config_names = 'config_codellama_Q'  # Modify this if you want to use another default config

print("SELECTED CONFIG FILES: " + config_names) 
configs.append(OmegaConf.load('configs/base_config.yaml'))

## Tener en cuenta que el ultimo tiene preferencia.

if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(f'configs/my_project/{config_name.strip()}.yaml'))

if enable_models:
    print("LOADING MODEL: ENABLED")
else: 
    print("LOADING MODEL: DISABLED")
    configs.append(OmegaConf.load(f'configs/my_project/disable_models.yaml'))

if "manual_long" in locals():
    configs.append(manual_long)

if "manual_config" in locals():
    configs.append(manual_config)

if "manual_batch" in locals():
    configs.append(manual_batch)

if "manual_temp" in locals():
    configs.append(manual_temp)

if "manual_adapter" in locals():
    configs.append(manual_adapter)


if mult:
    man_mult = OmegaConf.create({
                        "multiprocessing" : True
                    })

    configs.append(man_mult)

# else:
#     # The default
#     config_names = os.getenv('CONFIG_NAMES', None)
#     if config_names is None:
#         config_names = 'my_config'  # Modify this if you want to use another default config

#     configs = [OmegaConf.load('configs/base_config.yaml')]

#     if config_names is not None:
#         for config_name in config_names.split(','):
#             configs.append(OmegaConf.load(f'configs/{config_name.strip()}.yaml'))

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.merge(*configs)
logging.info(config)