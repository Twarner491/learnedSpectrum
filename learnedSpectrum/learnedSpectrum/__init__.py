"""
LearnedSpectrum: fMRI Learning Stage Classification with Vision Transformers
"""

__version__ = "0.1.0"

from .config import Config, DataConfig
from .data import (
    DatasetManager,
    FMRIDataset,
    BIDSManager,
    create_dataloaders
)
from .utils import (
    print_gpu_memory,
    get_optimizer,
    verify_model_devices,
    pretrain_transform,
    mixup
)

__all__ = [
    'Config',
    'DataConfig',
    'DatasetManager',
    'FMRIDataset', 
    'BIDSManager',
    'create_dataloaders',
    'print_gpu_memory',
    'get_optimizer',
    'verify_model_devices',
    'pretrain_transform',
    'mixup'
]
