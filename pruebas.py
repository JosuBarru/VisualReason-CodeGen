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
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper')
sys.path.append(script_dir)


from datasets.gqa import GQADataset


da = GQADataset(split="testdev", data_path='./data/gqa')
print(len(da))