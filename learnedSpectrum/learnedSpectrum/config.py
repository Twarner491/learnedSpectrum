"""
Configuration classes for LearnedSpectrum
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

@dataclass
class Config:
    # paths
    ROOT: str = str(Path(__file__).parent.parent.parent)
    CACHE: str = str(Path(ROOT) / "data/cleaned")
    CKPT_DIR: str = str(Path(ROOT) / "models/checkpoints")

    # training dynamics
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    USE_AMP: bool = True
    GRADIENT_ACCUMULATION_STEPS: int = 4

    # optimization
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.1
    NUM_EPOCHS: int = 30
    GRAD_CLIP: float = 0.5
    WARMUP_EPOCHS: int = 3
    MIN_LR: float = 1e-6

    # architecture
    VOLUME_SIZE: Tuple[int, int, int] = (64, 64, 30)
    PATCH_SIZE: int = 8
    EMBED_DIM: int = 384
    DEPTH: int = 12
    NUM_HEADS: int = 6
    MLP_RATIO: float = 4.0
    QKV_BIAS: bool = True
    DROP_RATE: float = 0.0
    ATTN_DROP_RATE: float = 0.0
    DROP_PATH_RATE: float = 0.1
    NUM_CLASSES: int = 4

    # augmentation
    MIXUP_ALPHA: float = 0.8
    CUTMIX_ALPHA: float = 1.0
    MIXUP_PROB: float = 1.0
    CUTMIX_PROB: float = 0.0
    MIXUP_SWITCH_PROB: float = 0.5
    MIXUP_MODE: str = "batch"
    LABEL_SMOOTHING: float = 0.1

@dataclass
class DataConfig:
    # Dataset URLs and paths
    DATASET_URLS: Dict[str, str] = field(default_factory=lambda: {
        'ds000002': 'https://openneuro.org/datasets/ds000002/versions/00002/download',
        'ds000011': 'https://openneuro.org/datasets/ds000011/versions/00001/download',
        'ds000017': 'https://openneuro.org/datasets/ds000017/versions/00001/download',
        'ds000052': 'https://openneuro.org/datasets/ds000052/versions/00001/download'
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
