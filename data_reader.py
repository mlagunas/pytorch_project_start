import warnings

import numpy as np
from os import path as paths
from torch.utils.data.dataset import Dataset
import torch

try:
    import accimage
except ImportError:
    accimage = None
    from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray)


class YourDataset(Dataset):
    def __init__(self, dir, trf_input=None, trf_output=None, is_train=True, ):
        # get train or test path accordingly
        dir = paths.join(dir, 'train') if is_train else paths.join(dir, 'test')

        # store transforms
        self.trf_input = trf_input
        self.trf_output = trf_output

        ## TODO: warninginit does not load any paths:
        warnings.warn('init does not load any paths')

    def __getitem__(self, index):
        ## TODO: warning: 
        warnings.warn('getitem not implemented')

    def __len__(self):
        ## TODO: warning: len not implemented
        warnings.warn('len not implemented')
