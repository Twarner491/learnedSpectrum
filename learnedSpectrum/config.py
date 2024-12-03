"""
Configuration classes for LearnedSpectrum
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch

@dataclass
class Config:

    ROOT: str = str(Path(__file__).parent.parent.parent)
    CACHE: str = str(Path(ROOT) / "data/cleaned")
    CKPT_DIR: str = str(Path(ROOT) / "models/checkpoints")

    # Device settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True  # Keep mixed precision for 4070 Ti
    
    # Training settings
    BATCH_SIZE: int = 16  # Reduced from 32/64
    NUM_WORKERS: int = 4  # Reduced from 8
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    
    # Model architecture
    PATCH_SIZE: int = 8
    EMBED_DIM: int = 384  # Reduced from 768
    DEPTH: int = 8  # Reduced from 12
    NUM_HEADS: int = 6  # Reduced from 12
    MLP_RATIO: float = 4.0
    NUM_CLASSES: int = 4
    
    # Regularization
    DROP_RATE: float = 0.1
    ATTN_DROP_RATE: float = 0.1
    DROP_PATH_RATE: float = 0.1
    WEIGHT_DECAY: float = 0.01
    
    # Optimization
    LEARNING_RATE: float = 2e-4
    WARMUP_EPOCHS: int = 5
    NUM_EPOCHS: int = 30
    GRADIENT_ACCUMULATION_STEPS: int = 2  # Added to compensate for smaller batch size
    
    # Volume dimensions
    VOLUME_SIZE: Tuple[int, int, int] = (64, 64, 30)

    GRADIENT_CLIP: float = 1.0
    LABEL_SMOOTHING: float = 0.1
    MIXUP_ALPHA: float = 0.2
    
    # Paths
    CKPT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"

    @property  
    def device(self):
        return torch.device(self.DEVICE)

def get_ds000002_stage(f):
    return 0.25 if 'run-2' in str(f) else 0.0

def get_ds000011_stage(f):
    return 0.5 if 'run-2' in str(f) else 0.25

def get_ds000017_stage(f):
    return 0.75 if 'run-2' in str(f) else 0.5

def get_ds000052_stage(f):
    if 'reversal' in str(f) and 'run-2' in str(f):
        return 1.0
    elif 'reversal' in str(f):
        return 0.67
    elif 'run-2' in str(f):
        return 0.33
    return 0.0

@dataclass
class DataConfig:
    # Dataset URLs and paths
    DATASET_URLS: Dict[str, str] = field(default_factory=lambda: {
        'ds000002': {
            'url': 'https://s3.amazonaws.com/openneuro/ds000002/ds000002_R2.0.5/compressed/ds000002_R2.0.5_raw.zip',
            'description': 'Classification Learning',
            'tr': 2.0,
            'stage_map': get_ds000002_stage
        },
        'ds000011': {
            'url': 'https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/compressed/ds000011_R2.0.1_raw.zip',
            'description': 'Mixed-gambles Task',
            'tr': 1.5,
            'stage_map': get_ds000011_stage
        },
        'ds000017': {
            'url': 'https://s3.amazonaws.com/openneuro/ds000017/ds000017_R2.0.1/compressed/ds000017_R2.0.1.zip',
            'description': 'Classification Learning and Reversal',
            'tr': 2.5,
            'stage_map': get_ds000017_stage
        },
        'ds000052': {
            'url': 'https://s3.amazonaws.com/openneuro/ds000052/ds000052_R2.0.0/compressed/ds052_R2.0.0_01-14.tgz',
            'description': 'Classification Learning and Stop-signal',
            'tr': 2.0,
            'stage_map': get_ds000052_stage
        }
    })
    
    # Data processing
    TARGET_SHAPE: Tuple[int, int, int] = (64, 64, 30)
    NORMALIZE: bool = True
    USE_WAVELET: bool = True
    WAVELET_NAME: str = 'db1'
    DECOMPOSITION_LEVEL: int = 3
    
    # Data splits
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1
    
    # Cache settings
    USE_CACHE: bool = True
    CACHE_DIR: str = "/content/fmri_cache"
    
    # BIDS configuration
    TASK_NAME: str = "learning"
    SPACE: str = "MNI152NLin2009cAsym"
    DESC: str = "preproc"