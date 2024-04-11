import torch
import numpy as np


def source_transform(array: np.ndarray) -> torch.Tensor:
    _tensor = torch.from_numpy(array)
    _tensor = _tensor.type(torch.FloatTensor)
    # _tensor = _tensor / 127.5 - 1
    _tensor = _tensor / 255
    return _tensor


def target_transform(array: np.ndarray) -> torch.Tensor:
    _tensor = torch.from_numpy(array)
    _tensor = _tensor.type(torch.FloatTensor)
    return _tensor
