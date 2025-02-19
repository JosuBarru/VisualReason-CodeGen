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
train = os.getenv('TRAIN', True) #  por defecto true porque va a ser lo que más usemos
train = train.lower() == 'true'
code = os.getenv('CODE', None)
num_inst_str = os.getenv('NUMINST')  # Defaults to None if the variable is not set
num_inst = int(num_inst_str) if num_inst_str not in (None, "") else None
batch_size = os.getenv('BATCHSIZE')
batch_size = int(batch_size) if batch_size is not None else None

config_names = []


model_configs = {
    "codellama": "config_codellama",
    "llama31Q": "config_codex_llama3.1-8b",
    "llama33Q": "config_codex_llama3.3-70b",
    "deepseek-qwen7b": "config_codex_deepseek-Qwen-7b",
    "deepseek-llama8b": "config_codex_deepseek-llama-8b",
    "deepseek-llama70b": "config_codex_deepseek-llama-70b"
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
                if code is not None:
                    manual_config = OmegaConf.create({
                        "use_cached_codex": True, 
                        "cached_codex_path": os.path.join('results/gqa/codex_results/', 'train' if train else 'testdev', code)
                    })
                
                else:
                    raise UserWarning(f"Add input file")

            elif execution_mode == 'codex':
                if codex_model in model_configs:
                    config_names.append(model_configs[codex_model])
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
            
            
            if train:
                config_names.append(dataset_name + '/' + 'train') 
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