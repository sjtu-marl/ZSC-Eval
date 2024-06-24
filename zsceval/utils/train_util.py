import random
import socket
from os import path as osp

import numpy as np
import torch


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_run_dir() -> str:
    socket.gethostname()
    base = osp.join(osp.expanduser("~"), "ZSC/results")
    return base
