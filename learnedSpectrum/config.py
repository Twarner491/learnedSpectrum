from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from .data import CognitiveProcess, TaskPhase
from .utils import (
    get_ds000002_stage,
    get_ds000011_stage,
    get_ds000017_stage,
    get_ds000052_stage
)


@dataclass
class Config:
    def __init__(self,
                 VOLUME_SIZE=(91, 109, 91),
                 PATCH_SIZE=16,
                 NUM_CLASSES=4,
                 BATCH_SIZE=8,
                 NUM_WORKERS=2,
                 LEARNING_RATE=2e-4,
                 NUM_EPOCHS=50,
                 NUM_HEADS=8,
                 DEVICE="cuda",
                 TEMPORAL_ANALYSIS=True,
                 CAUSAL_INFERENCE=True,
                 TEMPORAL_WEIGHT=0.5,
                 CAUSAL_WEIGHT=0.3,
                 ROOT=None):
        
        # Set VOLUME_SIZE first before post_init
        self.VOLUME_SIZE = VOLUME_SIZE
        
        if ROOT is None:
            ROOT = str(Path(__file__).parent.parent)
            
        # Base paths
        self.ROOT = ROOT
        self.CACHE = str(Path(self.ROOT) / "data/cleaned")
        self.CKPT_DIR = str(Path(self.ROOT) / "notebooks/models")
        self.VIS_DIR = str(Path(self.ROOT) / "visualizations")

        # Device and precision settings
        self.DEVICE = DEVICE
        self.USE_AMP = True
        self.FP16_OPT_LEVEL = "O2"
        self.AMP_DTYPE = torch.float16
        
        # Training settings
        self.BATCH_SIZE = 8  # Increased from 4
        self.NUM_WORKERS = 4  # Increased from 2
        self.PIN_MEMORY = True
        self.PERSISTENT_WORKERS = True
        self.PREFETCH_FACTOR = 4  # Increased from 2
        self.DROP_LAST = True  # Add this to prevent small last batches
        
        # Model architecture
        self.PATCH_SIZE = PATCH_SIZE
        self.EMBED_DIM = 192  # Reduced to match temporal states
        self.DEPTH = 6
        self.NUM_HEADS = NUM_HEADS
        self.MLP_RATIO = 4.0
        self.NUM_CLASSES = NUM_CLASSES
        
        # Regularization
        self.DROP_RATE = 0.1
        self.ATTN_DROP_RATE = 0.1
        self.DROP_PATH_RATE = 0.1
        self.WEIGHT_DECAY = 0.01
        self.STOCHASTIC_DEPTH_RATE = 0.1
        self.LABEL_SMOOTHING = 0.1
        
        # Optimization
        self.LEARNING_RATE = LEARNING_RATE
        self.WARMUP_EPOCHS = 5
        self.NUM_EPOCHS = NUM_EPOCHS
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.MIN_LR_RATIO = 0.01
        self.WARMUP_LR_RATIO = 0.05
        self.GRADIENT_CLIP_VAL = 1.0

        # Temporal settings
        self.TEMPORAL_ANALYSIS = TEMPORAL_ANALYSIS
        self.TEMPORAL_DIM = 192  # Match EMBED_DIM
        self.TEMPORAL_CHUNK_SIZE = 16
        self.TEMPORAL_DT = 0.1
        self.TEMPORAL_LAG = 2
        self.TEMPORAL_WEIGHT = TEMPORAL_WEIGHT
        self.ANALYSIS_WINDOW = 10
        self.MIN_STATE_DURATION = 0.5
        self.MAX_STATE_DURATION = 5.0

        # Causal settings
        self.CAUSAL_INFERENCE = CAUSAL_INFERENCE
        self.CAUSAL_WEIGHT = CAUSAL_WEIGHT
        self.CAUSAL_HIDDEN_DIM = 192  # Match EMBED_DIM
        self.PROPENSITY_WEIGHT = 0.1
        self.CAUSAL_KERNEL_SIZE = 3
        self.TREATMENT_SPARSITY = 0.1

        # LTC settings
        self.LTC_MIN_TAU = 1.0
        self.LTC_MAX_TAU = 100.0
        self.LTC_HIDDEN_DIM = 192  # Match EMBED_DIM
        self.LTC_NUM_CELLS = 8
        self.LTC_DT = 0.1

        # HJB settings
        self.HJB_WEIGHT = 0.3
        self.HJB_GAMMA = 0.99
        self.HJB_HIDDEN_DIM = 192  # Match EMBED_DIM
        self.HJB_LEARNING_RATE = 1e-4
        self.HJB_UPDATE_FREQ = 5
        self.VALUE_COEF = 0.5

        # Memory management
        self.PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
        
        # Call post_init after all attributes are set
        self.__post_init__()
        
    def __post_init__(self):
        # Validate dimensions
        z_dim = self.VOLUME_SIZE[2]
        pad_size = (self.NUM_HEADS * 2) - (z_dim % (self.NUM_HEADS * 2))
        if pad_size != self.NUM_HEADS * 2:
            self.VOLUME_SIZE = (
                self.VOLUME_SIZE[0],
                self.VOLUME_SIZE[1],
                z_dim + pad_size
            )
        
        # Validate settings
        assert self.BATCH_SIZE % self.GRADIENT_ACCUMULATION_STEPS == 0
        assert 0 <= self.LABEL_SMOOTHING <= 1
        assert self.WARMUP_EPOCHS < self.NUM_EPOCHS
        assert self.PATCH_SIZE <= min(self.VOLUME_SIZE[:2])
        assert self.VOLUME_SIZE[2] % 2 == 0

    @property
    def device(self):
        return torch.device(self.DEVICE)


@dataclass
class DataConfig:
    def __init__(self, 
                 USE_CACHE=True,
                 NORMALIZE=True,
                 USE_WAVELET=True,
                 TRAIN_SPLIT=0.7,
                 VAL_SPLIT=0.15,
                 CACHE_DIR="../data/processed",
                 TARGET_SHAPE=(91, 109, 91)):
        self.USE_CACHE = USE_CACHE
        self.NORMALIZE = NORMALIZE
        self.USE_WAVELET = USE_WAVELET
        self.TRAIN_SPLIT = TRAIN_SPLIT
        self.VAL_SPLIT = VAL_SPLIT
        self.CACHE_DIR = CACHE_DIR
        self.TARGET_SHAPE = TARGET_SHAPE

        self.DATASET_URLS = {
            'ds000002': {
                'url': 'https://s3.amazonaws.com/openneuro/ds000002/ds000002_R2.0.5/compressed/ds000002_R2.0.5_raw.zip',
                'description': 'Classification Learning',
                'tr': 2.0,
                'stage_map': get_ds000002_stage
            },
            'ds000011': {
                'url': 'https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/compressed/ds000011_R2.0.1_raw.zip',
                'description': 'Mixed-gambles Task',
                'tr': 2.0,
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
        }

        self.TARGET_SHAPE: Tuple[int, int, int] = (64, 64, 30)
        self.NORMALIZE: bool = True
        self.USE_WAVELET: bool = True
        self.WAVELET_NAME: str = 'db1'
        self.DECOMPOSITION_LEVEL: int = 2
        self.SPATIAL_SMOOTHING_SIGMA: float = 1.5
        self.TEMPORAL_FILTER_CUTOFF: float = 0.1

        self.TRAIN_SPLIT: float = 0.7
        self.VAL_SPLIT: float = 0.15
        self.TEST_SPLIT: float = 0.15

        self.USE_CACHE: bool = True
        self.CACHE_DIR: str = str(Path("../data/processed").resolve())
        self.CACHE_SIZE_LIMIT_GB: float = 32.0

        self.TASK_NAME: str = "learning"
        self.SPACE: str = "MNI152NLin2009cAsym"
        self.DESC: str = "preproc"
        
        self.BALANCE_CLASSES: bool = True
        self.USE_METADATA: bool = True

        self.METADATA_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
            'learning_phase': r'phase-(\w+)',
            'cognitive_process': r'process-(\w+)',
        })
        
    def __post_init__(self):
        assert sum([self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT]) == 1.0
        assert all(0 < x < 1 for x in [self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT])