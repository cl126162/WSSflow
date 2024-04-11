import os
import random
from typing import Optional, Union, Callable, Tuple

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class PatchedDataset(Dataset):
    def __init__(
            self,
            offset,
            shift,
            fullFrame,
            root: Union[os.PathLike, str, bytes],
            source_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ):
        super().__init__()
        self._offset = offset
        self._shift = shift
        self._fullFrame = fullFrame
        self._file_path = root
        self.source_transform = source_transform
        self.target_transform = target_transform

        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Couldn't locate source file under {self._file_path}")

        # placeholder for file handle, extract file length
        self._file = None  # must bet set in __getitem__ https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
        with h5py.File(name=self._file_path, mode="r") as f:
            self._length = len(f['Flow'])

    def __len__(self):
        return self._length

    def __getitem__(self, index):

        if self._file is None:  # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
            self._file = h5py.File(name=self._file_path, mode="r")
            self._source = np.empty(shape=self._file['Images'].shape, dtype=np.uint8)
            self._target = np.empty(shape=self._file['Flow'].shape, dtype=np.float32)
            self._source[:] = self._file['Images'][:]
            self._target[:] = self._file['Flow'][:]

        # get data slices, transform in pipeline
        source, target = self._source[index], self._target[index]
        if self.source_transform is not None:
            source = self.source_transform(source)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self._fullFrame:
            return source, target
        else:
            source = torch.unsqueeze(source, dim=0)
            target = torch.unsqueeze(target, dim=0)

            # create patches of image and flow
            patches = source.unfold(3, self._offset, self._shift)\
                            .unfold(2, self._offset, self._shift)\
                            .permute(0, 2, 3, 1, 5, 4)
            patches = patches.reshape((-1, 2, self._offset, self._offset))
            flow_patches = target.unfold(3, self._offset, self._shift)\
                                 .unfold(2, self._offset, self._shift)\
                                 .permute(0, 2, 3, 1, 5, 4)
            flow_patches = flow_patches.reshape((-1, 2, self._offset, self._offset))
            return patches, flow_patches

class testing_dataset(Dataset):
    def __init__(
            self,
            root: Union[os.PathLike, str, bytes],
            source_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ):
        super().__init__()
        self._file_path = root
        self.source_transform = source_transform
        self.target_transform = target_transform

        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Couldn't locate source file under {self._file_path}")

        # placeholder for file handle, extract file length
        self._file = None  # must bet set in __getitem__ https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
        with h5py.File(name=self._file_path, mode="r") as f:
            self._length = len(f['Flow'])

    def __len__(self):
        return self._length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # get file handle for local worker
        if self._file is None:  # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
            self._file = h5py.File(name=self._file_path, mode="r")
            self._source = np.empty(shape=self._file['Images'].shape, dtype=np.uint8)
            self._target = np.empty(shape=self._file['Flow'].shape, dtype=np.float32)

            self._source[:] = self._file['Images'][:]
            self._target[:] = self._file['Flow'][:]


        # get data slices, transform in pipeline
        source, target= self._source[index], self._target[index]

        if self.source_transform is not None:
            source = self.source_transform(source)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return source, target

