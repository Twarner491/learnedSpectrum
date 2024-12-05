from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch


@dataclass
class Config:
    # paths
    ROOT: str = str(Path(__file__).parent.parent)  
    CACHE: str = str(Path(ROOT) / "data/cleaned")
    CKPT_DIR: str = str(Path(ROOT) / "notebooks/models")

    # compute config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True  # mixed precision training
    FP16_OPT_LEVEL: str = "O2"  # aggressive mixed precision
    
    # data loading
    BATCH_SIZE: int = 8  # Reduced from 16
    NUM_WORKERS: int = 2  # Reduced from 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    
    # architecture (vit-tiny optimal for small datasets)
    PATCH_SIZE: int = 8   # smaller patches = finer detail
    EMBED_DIM: int = 240  # wider network
    DEPTH: int = 8        # deeper
    NUM_HEADS: int = 5    # more heads
    MLP_RATIO: float = 4.0  # wider ffn
    NUM_CLASSES: int = 4
    
    # regularization (cranked up bc small dataset)
    DROP_RATE: float = 0.4
    ATTN_DROP_RATE: float = 0.2
    DROP_PATH_RATE: float = 0.2
    WEIGHT_DECAY: float = 0.3
    STOCHASTIC_DEPTH_RATE: float = 0.2
    
    # training dynamics
    LEARNING_RATE: float = 1e-5  # decreased w/ larger bs
    WARMUP_EPOCHS: int = 10
    NUM_EPOCHS: int = 20
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MIN_LR_RATIO: float = 0.01
    WARMUP_LR_RATIO: float = 0.01
    
    # data augmentation
    VOLUME_SIZE: Tuple[int, int, int] = (64, 64, 30)
    GRADIENT_CLIP: float = 1.0
    LABEL_SMOOTHING: float = 0.15  # reduced
    MIXUP_ALPHA: float = 0.9  # increased
    CUTMIX_ALPHA: float = 1.0
    CUTMIX_MINMAX: Tuple[float, float] = (0.3, 0.7)
    RandAugment_N: int = 3  # num transforms
    RandAugment_M: int = 12  # severity
    
    # loss weights
    CE_WEIGHT: float = 1.0
    CONSISTENCY_WEIGHT: float = 0.1
    CONTRASTIVE_WEIGHT: float = 0.05
    
    # validation
    EARLY_STOPPING_PATIENCE: int = 15
    VALIDATION_FREQ: int = 1
    SAVE_FREQ: int = 5
    
    @property
    def device(self):
        return torch.device(self.DEVICE)

    def __post_init__(self):
        # validation
        assert self.BATCH_SIZE % self.GRADIENT_ACCUMULATION_STEPS == 0
        assert self.VOLUME_SIZE[2] % (self.NUM_HEADS * 2) == 0
        assert 0 <= self.LABEL_SMOOTHING <= 1
        assert self.WARMUP_EPOCHS < self.NUM_EPOCHS
        
        # critical for fmri
        assert self.PATCH_SIZE <= min(self.VOLUME_SIZE[:2])
        assert self.VOLUME_SIZE[2] % 2 == 0  # even temporal dim


@dataclass
class DataConfig:
    # keep your existing dataset configs
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
    
    # enhanced preprocessing
    TARGET_SHAPE: Tuple[int, int, int] = (64, 64, 30)
    NORMALIZE: bool = True
    USE_WAVELET: bool = True
    WAVELET_NAME: str = 'db1'
    DECOMPOSITION_LEVEL: int = 2  # increased
    SPATIAL_SMOOTHING_SIGMA: float = 1.5
    TEMPORAL_FILTER_CUTOFF: float = 0.1
    
    # robust splits
    TRAIN_SPLIT: float = 0.7  # reduced
    VAL_SPLIT: float = 0.15   # increased
    TEST_SPLIT: float = 0.15  # increased
    
    # caching
    USE_CACHE: bool = True
    CACHE_DIR: str = str(Path("data/processed").resolve())
    CACHE_SIZE_LIMIT_GB: float = 32.0
    
    # bids
    TASK_NAME: str = "learning"
    SPACE: str = "MNI152NLin2009cAsym"
    DESC: str = "preproc"
    
    def __post_init__(self):
        assert sum([self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT]) == 1.0
        assert all(0 < x < 1 for x in [self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT])


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