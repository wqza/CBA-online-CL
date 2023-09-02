# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import numpy as np


def get_device(gpu_id=None) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    if gpu_id is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cuda:{}".format(gpu_id))


def data_path() -> str:
    """
    Returns the data path.
    """
    return '/home/iid/WQZA/dataset/'
    # return 'E://dataset/'


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return '/home/iid/WQZA/code/DER/'
    # return './'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
