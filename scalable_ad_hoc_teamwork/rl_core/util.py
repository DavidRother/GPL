import random
import numpy as np
import torch
import os


def make_deterministic(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def make_deterministic_old(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def clean_dict_for_evalpy(stat_dict):
    for key, value in stat_dict.items():
        if isinstance(value, np.ndarray) and value.size() == 1:
            stat_dict[key] = value.item()
        if isinstance(value, np.float64):
            stat_dict[key] = float(value)
        if isinstance(value, np.integer):
            stat_dict[key] = int(value)
    return stat_dict


def info_callback_func(policy, info):
    if info["recipe_done"]:
        policy.switch_active_tasks([])


