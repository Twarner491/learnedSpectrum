import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates
from scipy.stats import special_ortho_group
import pywt
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum, auto, IntEnum
from dataclasses import dataclass
import re
from tqdm.auto import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class CognitiveProcess(IntEnum):
    ACQUISITION = 0
    CONSOLIDATION = 1
    TRANSFER = 2
    REVERSAL = 3


class TaskPhase(IntEnum):
    EARLY = 4
    MIDDLE = 5
    LATE = 6
    MASTERY = 7


@dataclass
class TaskMetadata:
    process: CognitiveProcess
    phase: TaskPhase
    tr: float
    multiband: Optional[int] = None
    feedback: bool = False


class AugmentationBase:
    """stochastic augmentation base. enforces invertibility constraints."""
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x if random.random() > self.p else self.forward(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class TemporalMask(AugmentationBase):
    def __init__(self, max_mask_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_mask_size = max_mask_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_size = random.randint(1, self.max_mask_size)
        start = random.randint(0, x.shape[-2] - mask_size)
        x = x.clone()
        x[..., start:start+mask_size, :] = 0
        return x
    

class SpatialMask(AugmentationBase):
    def __init__(self, mask_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask
    

class GaussianNoise(AugmentationBase):
    def __init__(self, std: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std
    

class IntensityShift(AugmentationBase):
    def __init__(self, max_shift: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.uniform(-self.max_shift, self.max_shift)
        return x + shift
    

class RandomFlip(AugmentationBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [random.randint(0,1)])
    

class RandomRotate(AugmentationBase):
    def __init__(self, max_angle: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_angle = max_angle
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = random.randint(-self.max_angle, self.max_angle)
        return torch.from_numpy(
            rotate(x.numpy(), angle, axes=(0,1), reshape=False)
        )
    

class ElasticDeform(AugmentationBase):
    def __init__(self, alpha: float = 1000, sigma: float = 50, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.sigma = sigma
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(x):
            x_np = x.numpy()
        else:
            x_np = x
            
        # Get spatial dimensions only (ignore batch/channel dims)
        spatial_shape = x_np.shape[-2:]  # Assuming (H, W) are last two dimensions
        
        # Generate displacement fields for spatial dimensions only
        dx = gaussian_filter(np.random.randn(*spatial_shape), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(*spatial_shape), self.sigma) * self.alpha
        
        x_mesh, y_mesh = np.meshgrid(
            np.arange(spatial_shape[1]), 
            np.arange(spatial_shape[0])
        )
        
        indices = (
            np.reshape(y_mesh + dy, (-1, 1)),
            np.reshape(x_mesh + dx, (-1, 1))
        )
        
        # Apply deformation to each channel/time point independently
        if x_np.ndim > 2:
            result = np.stack([
                map_coordinates(slice_2d, indices, order=1).reshape(spatial_shape)
                for slice_2d in x_np.reshape(-1, *spatial_shape)
            ])
            result = result.reshape(x_np.shape)
        else:
            result = map_coordinates(x_np, indices, order=1).reshape(spatial_shape)
            
        return torch.from_numpy(result.copy())
    

class BIDSValidator:
    def __init__(self):
        self.patterns = {
            'func': r'sub-\d+(?:_ses-\w+)?_task-\w+_(?:acq-\w+_)?(?:rec-\w+_)?(?:run-\d+_)?(?:echo-\d+_)?bold\.nii(?:\.gz)?'
        }

    def is_valid_bids(self, filepath: Path) -> bool:
        import re
        return bool(re.match(self.patterns['func'], filepath.name))
    

class FMRIAugmentor:
    """hardcore data aug suite for limited neuroimaging samples"""
    def __init__(self, p: float = 0.5):
        self.augmentations = [
            TemporalMask(p=p),
            SpatialMask(p=p),
            GaussianNoise(p=p),
            IntensityShift(p=p),
            RandomFlip(p=p),
            RandomRotate(p=p),
            ElasticDeform(p=p)
        ]
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x)
        return x
    

def sanitized_collate(batch):
    """robust batch assembly w/ corrupt sample handling"""
    valid_batch = [(x, y) for x, y in batch if x is not None]
    if not valid_batch:
        raise RuntimeError("entire batch corrupt")
    return torch.utils.data.dataloader.default_collate(valid_batch)


class BIDSManager:
    def __init__(self, data_config):
        self.config = data_config
        self.raw_dir = Path("../data/raw").absolute()
        self.processed_dir = Path("../data/processed").absolute()
        self.valid_files = set()
        self._init_dirs()

    def _init_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def validate_nifti(self, path: Path) -> bool:
        """strict nifti validation"""
        try:
            with open(path, 'rb') as f:
                if f.read(2) != b'\x1f\x8b': return False
            img = nib.load(str(path))
            if img.header['sizeof_hdr'] != 348: return False
            shape = img.header.get_data_shape()
            if not (3 <= len(shape) <= 4): return False
            if "Classificationprobewithoutfeedback" in str(path): return False
            return True
        except Exception:
            return False

    def _find_dataset_root(self, dataset_id: str) -> Path:
        patterns = [
            f"{dataset_id}",
            f"{dataset_id}_R*.0.*",
            f"ds{dataset_id.split('ds')[1]}_R*.0.*",
            "*/*bold.nii.gz"
        ]
        for pattern in patterns:
            candidates = list(self.raw_dir.rglob(pattern))
            if candidates:
                valid = [p for p in candidates if "func" in str(p.parent)]
                if valid: return valid[0].parent.parent
                return candidates[0]
        raise FileNotFoundError(f"dataset {dataset_id} not found")

    def download_dataset(self, dataset_id: str) -> Path:
        try:
            return self._find_dataset_root(dataset_id)
        except Exception as e:
            logger.error(f"failed to locate {dataset_id}: {e}")
            raise


class NiftiLoader:
    def __init__(self, config, bids_manager):
        self.config = config
        self.bids_manager = bids_manager
        self.validator = BIDSValidator()
        self._validated_cache = set()
        self._data_cache = {}
        self.error_log = Path("nifti_errors.log")
        self.error_count = 0
        self.max_errors = 100
        self.cache_dir = Path("../data/processed")
        self.cache_dir.mkdir(exist_ok=True)
        self._init_logging()

    def _init_logging(self):
        handler = logging.FileHandler(self.error_log)
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _parallel_validate(self, paths: List[Path], max_workers: int = 8) -> List[Path]:
        valid_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.validate_nifti, p): p for p in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc='validating'):
                path = futures[future]
                try:
                    if future.result():
                        valid_paths.append(path)
                        self._validated_cache.add(path)
                except Exception as e:
                    logger.error(f"validation fail: {path} - {e}")
        return valid_paths

    @lru_cache(maxsize=1024)
    def validate_nifti(self, path: Path) -> bool:
        if path in self._validated_cache:
            return True
            
        try:
            if not path.exists():
                return False

            cache_path = self.cache_dir / f"{path.stem}.npy"
            if cache_path.exists():
                return True

            with open(path, 'rb') as f:
                if f.read(2) != b'\x1f\x8b':
                    return False

            img = nib.load(str(path))
            hdr = img.header
            
            if hdr['sizeof_hdr'] != 348:
                return False
                
            shape = hdr.get_data_shape()
            pixdim = hdr.get_zooms()
            
            if not (3 <= len(shape) <= 4):
                return False
                
            if any(d <= 0 for d in pixdim[:3]):
                return False
                
            if "probe" in str(path).lower():
                return False

            current_shape = shape[:3]
            if current_shape != self.config.TARGET_SHAPE:
                factors = [t/c for t,c in zip(self.config.TARGET_SHAPE, current_shape)]
                try:
                    data = img.get_fdata()
                    zoom(data[...,0] if len(shape) == 4 else data, factors, order=3)
                except Exception:
                    return False

            return True

        except Exception as e:
            logger.error(f"validation fail: {path} - {e}")
            return False

    def _parallel_preprocess(self, paths: List[Path], max_workers: int = 4):
        """Process files in parallel with better error handling and cache checking"""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        def process_single_file(path: Path) -> Optional[Tuple[Path, np.ndarray]]:
            try:
                # Determine output path first
                rel_path = path.relative_to(self.bids_manager.raw_dir)
                out_path = self.bids_manager.processed_dir / f"{rel_path.stem}.npy"
                
                # Skip if already processed
                if out_path.exists():
                    logger.debug(f"Skipping {path.name} - already processed")
                    return out_path
                    
                # Create output directory if needed
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process file
                img = nib.load(str(path), mmap=True)
                data = np.asanyarray(img.dataobj)
                
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
                
                processed = self._preprocess_volume(data)
                np.save(out_path, processed.astype(np.float32))
                
                logger.debug(f"Processed {path.name} -> {out_path}")
                return out_path
                
            except Exception as e:
                logger.error(f"preprocess fail: {path} - {str(e)}")
                return None
        
        # Filter out already processed files
        to_process = []
        for path in paths:
            rel_path = path.relative_to(self.bids_manager.raw_dir)
            out_path = self.bids_manager.processed_dir / f"{rel_path.stem}.npy"
            if not out_path.exists():
                to_process.append(path)
        
        if not to_process:
            logger.info("All files already processed")
            return
            
        logger.info(f"Processing {len(to_process)} new files")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(process_single_file, to_process),
                total=len(to_process),
                desc="preprocessing"
            ))

    def _preprocess_volume(self, data: np.ndarray) -> np.ndarray:
        if data.ndim not in (3, 4):
            raise ValueError(f"invalid dims: {data.ndim}")
        
        # Convert float16 to float32 before processing
        if data.dtype == np.float16:
            data = data.astype(np.float32)
            
        if data.ndim == 3:
            data = data[..., np.newaxis]
            
        current_shape = data.shape[:3]
        target_shape = self.config.TARGET_SHAPE
        scale_factors = [t/c for t, c in zip(target_shape, current_shape)]
        
        # Process in chunks to avoid memory issues
        processed_data = []
        chunk_size = 10  # Process 10 timepoints at a time
        
        for t in range(0, data.shape[-1], chunk_size):
            chunk = data[..., t:t + chunk_size]
            # Apply spatial scaling to chunk
            scaled_chunk = zoom(chunk, scale_factors + [1], order=1)  # Reduced order for memory
            processed_data.append(scaled_chunk)
        
        data = np.concatenate(processed_data, axis=-1)
        
        if self.config.NORMALIZE:
            # Normalize each timepoint independently
            for t in range(data.shape[-1]):
                t_data = data[..., t]
                t_mean = t_data.mean()
                t_std = t_data.std() + 1e-6
                data[..., t] = (t_data - t_mean) / t_std
            
        return data

    def _wavelet_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.config.USE_WAVELET:
            return data
            
        padded = np.pad(data, ((0,0), (0,0), (0,2), (0,0)), mode='symmetric')
        
        try:
            coeffs = pywt.wavedec(padded, wavelet='haar', level=2, axis=-2)
            scaled_coeffs = [zoom(c, (1, 1, coeffs[0].shape[2]/c.shape[2], 1), order=1)
                           for c in coeffs]
            return np.concatenate(scaled_coeffs, axis=-1)
        except Exception as e:
            logger.warning(f"wavelet fail: {e}, using raw")
            return data

    @lru_cache(maxsize=256)
    def load_and_preprocess(self, filepath: Path) -> Optional[torch.Tensor]:
        cache_path = self.cache_dir / f"{filepath.stem}.npy"
        
        try:
            if cache_path.exists():
                # Load without mmap to ensure writeable array
                data = np.load(str(cache_path), mmap_mode=None)
            else:
                data = self._preprocess_single(filepath)
                
            if data is None:
                return None
                
            # Ensure data is writable and contiguous before tensor conversion
            data = np.array(data, copy=True)
            data = np.ascontiguousarray(data)
            return torch.from_numpy(data)

        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise RuntimeError(f"excessive fails: {self.error_count}")
            logger.error(f"load fail: {filepath} - {e}")
            return None
        
        
class FMRIDataset(Dataset):
    """Dataset class for fMRI data"""
    def __init__(self, data_paths, labels, transform=None, max_timepoints=200):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.max_timepoints = max_timepoints

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load data
        data_source = self.data_paths[idx]
        
        # Handle different data source types
        if isinstance(data_source, (str, Path)):
            data = np.load(data_source)
        elif isinstance(data_source, np.ndarray):
            data = data_source
        elif isinstance(data_source, np.memmap):
            data = np.array(data_source)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")
            
        # Handle time dimension
        if data.shape[-1] > self.max_timepoints:
            # Truncate to max_timepoints
            data = data[..., :self.max_timepoints]
        elif data.shape[-1] < self.max_timepoints:
            # Pad with zeros
            pad_width = [(0, 0)] * (data.ndim - 1) + [(0, self.max_timepoints - data.shape[-1])]
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
            
        label = self.labels[idx]
        
        # Apply transforms if any
        if self.transform:
            data = self.transform(data)
            
        return torch.from_numpy(data).float(), label


class DatasetManager:
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.cache_dir = Path(data_config.CACHE_DIR)
        self.data_paths = []
        
        # Map cognitive processes and phases to numerical labels
        self.label_mapping = {
            # Cognitive Process labels (0-3)
            'acquisition': CognitiveProcess.ACQUISITION,
            'consolidation': CognitiveProcess.CONSOLIDATION,
            'transfer': CognitiveProcess.TRANSFER,
            'reversal': CognitiveProcess.REVERSAL,
            
            # Task Phase labels (4-7)
            'early': TaskPhase.EARLY,
            'middle': TaskPhase.MIDDLE,
            'late': TaskPhase.LATE,
            'mastery': TaskPhase.MASTERY
        }
        
        # Mapping rules for task patterns to learning stages
        self.task_stage_mapping = {
            # Deterministic classification
            r'task-deterministicclassification_run-01': 'acquisition',
            r'task-deterministicclassification_run-02': 'consolidation',
            
            # Probabilistic classification
            r'task-probabilisticclassification_run-01': 'acquisition',
            r'task-probabilisticclassification_run-02': 'consolidation',
            
            # Mixed event related probe
            r'task-mixedeventrelatedprobe_run-01': 'acquisition',
            r'task-mixedeventrelatedprobe_run-02': 'transfer',
            
            # Dual task weather prediction
            r'task-Dualtaskweatherprediction_run-01': 'transfer',
            r'task-Dualtaskweatherprediction_run-02': 'transfer',
            
            # Single task weather prediction
            r'task-Singletaskweatherprediction_run-01': 'acquisition',
            r'task-Singletaskweatherprediction_run-02': 'consolidation',
            
            # Selective stop signal task
            r'task-selectivestopsignaltask_run-01': 'acquisition',
            r'task-selectivestopsignaltask_run-02': 'consolidation',
            r'task-selectivestopsignaltask_run-03': 'transfer',
            
            # Handle timepoint variations
            r'ses-timepoint1_task-probabilisticclassification_run-01': 'acquisition',
            r'ses-timepoint1_task-probabilisticclassification_run-02': 'consolidation',
            r'ses-timepoint2_task-probabilisticclassification_run-01': 'acquisition',
            r'ses-timepoint2_task-probabilisticclassification_run-02': 'consolidation',
            
            r'ses-timepoint1_task-selectivestopsignaltask_run-01': 'acquisition',
            r'ses-timepoint1_task-selectivestopsignaltask_run-02': 'consolidation',
            r'ses-timepoint1_task-selectivestopsignaltask_run-03': 'transfer',
            r'ses-timepoint2_task-selectivestopsignaltask_run-01': 'acquisition',
            r'ses-timepoint2_task-selectivestopsignaltask_run-02': 'consolidation',
            r'ses-timepoint2_task-selectivestopsignaltask_run-03': 'transfer',
            
            # Basic weather prediction
            r'task-weatherprediction_run-1': 'acquisition',
            r'task-weatherprediction_run-2': 'consolidation',
            
            # Reversal weather prediction
            r'task-reversalweatherprediction_run-1': 'reversal',
            r'task-reversalweatherprediction_run-2': 'reversal',
            
            # Tone counting
            r'task-tonecounting': 'transfer'
        }
        
        self._load_data_paths()
        
    def _create_dataset(self, paths, labels, transform=None):
        """Create dataset from paths and labels"""
        # Debug logging
        logger.info(f"Creating dataset with {len(paths)} paths and {len(labels)} labels")
        
        # Find maximum number of timepoints across all samples
        max_timepoints = 0
        for path in paths:
            try:
                data = np.load(path)
                if isinstance(data, np.ndarray):  # Verify it's a numpy array
                    max_timepoints = max(max_timepoints, data.shape[-1])
                else:
                    logger.warning(f"Loaded data is not a numpy array: {type(data)}")
            except Exception as e:
                logger.error(f"Error loading file {path}: {str(e)}")
                continue
            
        logger.info(f"Maximum timepoints found: {max_timepoints}")
        
        if max_timepoints == 0:
            logger.error("No valid timepoints found in any samples!")
            return None
        
        # Add missing patterns to task_stage_mapping
        self.task_stage_mapping.update({
            # Basic weather prediction
            r'task-weatherprediction_run-1': 'acquisition',
            r'task-weatherprediction_run-2': 'consolidation',
            
            # Reversal weather prediction
            r'task-reversalweatherprediction_run-1': 'reversal',
            r'task-reversalweatherprediction_run-2': 'reversal',
            
            # Tone counting
            r'task-tonecounting': 'transfer'
        })
        
        return SpectrogramDataset(
            paths=paths,
            labels=labels,
            max_timepoints=max_timepoints,
            transform=transform
        )
    
    def _load_data_paths(self):
        if self.cache_dir.exists():
            self.data_paths = list(self.cache_dir.rglob("*.npy"))
            logger.info(f"Found {len(self.data_paths)} cached samples in {self.cache_dir}")
        else:
            logger.warning(f"Cache directory {self.cache_dir} not found")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_paths.sort()
        self.chunk_size = 10
        self.max_memory_gb = self.data_config.CACHE_SIZE_LIMIT_GB

    def prepare_datasets(self):
        """Prepare train, validation and test datasets with balanced labels"""
        # Get valid samples and their labels
        valid_samples = []
        labels = []
        
        for path in self.data_paths:
            stage = self._determine_learning_stage(path)
            if stage:
                valid_samples.append(path)
                labels.append(self.label_mapping[stage])
        
        # Convert to numpy for easier manipulation
        samples = np.array(valid_samples)
        labels = np.array(labels)
        
        # Print label distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Label distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"Label {label}: {count} samples")
        
        # Stratified split
        # First split train and temp
        train_samples, temp_samples, train_labels, temp_labels = train_test_split(
            samples, labels,
            test_size=1-self.data_config.TRAIN_SPLIT,
            stratify=labels,
            random_state=42
        )
        
        # Then split temp into val and test
        val_ratio = self.data_config.VAL_SPLIT / (1-self.data_config.TRAIN_SPLIT)
        val_samples, test_samples, val_labels, test_labels = train_test_split(
            temp_samples, temp_labels,
            test_size=0.5,
            stratify=temp_labels,
            random_state=42
        )
        
        # Create datasets with balanced sampling
        train_ds = self._create_dataset(train_samples, train_labels)
        val_ds = self._create_dataset(val_samples, val_labels)
        test_ds = self._create_dataset(test_samples, test_labels)
        
        # Verify label distribution in splits
        for name, dataset in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
            labels = [label for _, label in dataset]
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"\n{name} set label distribution:")
            for label, count in zip(unique, counts):
                logger.info(f"Label {label}: {count} samples")
        
        return train_ds, val_ds, test_ds

    def _extract_label_from_path(self, path: Path) -> Optional[int]:
        """Extract learning stage label from path"""
        path_str = str(path).lower()
        
        # First try to match task stage patterns
        for pattern, stage in self.task_stage_mapping.items():
            if re.search(pattern, path_str):
                if stage in self.label_mapping:
                    return self.label_mapping[stage].value - 1  # Convert to 0-based index
                
        # If no stage pattern matched, try to infer from metadata if available
        try:
            metadata = self._extract_metadata(path)
            if metadata and metadata.process:
                return metadata.process.value - 1
            if metadata and metadata.phase:
                return metadata.phase.value - 1 + len(CognitiveProcess)  # Offset phase labels
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {path}: {e}")
            
        logger.warning(f"Could not determine learning stage for {path}")
        return None
        
    def _extract_metadata(self, path: Path) -> Optional[TaskMetadata]:
        """Extract task metadata from filename or associated metadata files"""
        try:
            # Try to find associated JSON sidecar (BIDS format)
            json_path = path.with_suffix('.json')
            if json_path.exists():
                import json
                with open(json_path) as f:
                    metadata = json.load(f)
                    
                # Extract learning stage info from metadata
                process = None
                phase = None
                
                # Example metadata parsing - adjust based on your metadata structure
                if 'LearningPhase' in metadata:
                    phase_str = metadata['LearningPhase'].lower()
                    if phase_str in ['early', 'initial']:
                        phase = TaskPhase.EARLY
                    elif phase_str in ['middle', 'intermediate']:
                        phase = TaskPhase.MIDDLE
                    elif phase_str in ['late', 'final']:
                        phase = TaskPhase.LATE
                    elif phase_str in ['mastery', 'expert']:
                        phase = TaskPhase.MASTERY
                        
                if 'CognitiveProcess' in metadata:
                    process_str = metadata['CognitiveProcess'].lower()
                    if process_str in ['acquisition', 'learning']:
                        process = CognitiveProcess.ACQUISITION
                    elif process_str in ['consolidation', 'retention']:
                        process = CognitiveProcess.CONSOLIDATION
                    elif process_str in ['transfer']:
                        process = CognitiveProcess.TRANSFER
                    elif process_str in ['reversal']:
                        process = CognitiveProcess.REVERSAL
                
                return TaskMetadata(
                    process=process,
                    phase=phase,
                    tr=metadata.get('RepetitionTime', None),
                    multiband=metadata.get('MultibandAccelerationFactor', None),
                    feedback=metadata.get('Feedback', False)
                )
                
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
            
        return None

    def _balance_samples(self, samples, label_counts):
        """Balance dataset by undersampling majority classes"""
        min_count = min(label_counts.values())
        balanced_samples = []
        
        # Group samples by label
        label_groups = {i: [] for i in range(len(self.label_mapping))}
        for sample in samples:
            label_groups[sample[1]].append(sample)
            
        # Undersample each group to min_count
        for label in label_groups:
            if len(label_groups[label]) > min_count:
                label_groups[label] = random.sample(label_groups[label], min_count)
            balanced_samples.extend(label_groups[label])
            
        return balanced_samples

    def _load_single_sample(self, path: Path) -> Optional[Tuple[np.ndarray, int]]:
        """Load and preprocess a single sample with memory management"""
        try:
            # Load data with reduced precision
            data = np.load(path, mmap_mode='r')  # Memory-mapped reading
            
            # Extract label from filename
            label = self._extract_label_from_path(path)
            
            # Process in smaller chunks if needed
            if data.nbytes > 1e9:  # 1GB threshold
                logger.warning(f"Large file detected: {path}")
                data = self._process_large_file(data)
            
            # Convert to float32 for memory efficiency
            data = data.astype(np.float32)
            
            return (data, label)
            
        except Exception as e:
            logger.warning(f"Sample loading failed {path}: {str(e)}")
            return None

    def _process_large_file(self, data):
        """Handle large files by processing in chunks"""
        chunk_size = 1000  # Adjust based on your memory constraints
        processed_chunks = []
        
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i + chunk_size].copy()
            # Process chunk here
            processed_chunks.append(chunk)
            
        return np.concatenate(processed_chunks)
    
    def _determine_learning_stage(self, path: Path) -> Optional[str]:
        """Determine learning stage from file path"""
        path_str = str(path)
        
        # Debug the incoming path
        logger.debug(f"Determining stage for path: {path_str}")
        
        for pattern, stage in self.task_stage_mapping.items():
            if re.search(pattern, path_str):
                logger.debug(f"Matched pattern '{pattern}' to stage '{stage}'")
                return stage
                
        logger.warning(f"Could not determine learning stage for {path}")
        return None
    
    
def pad_collate(batch):
    """Memory-efficient padding for temporal dimension."""
    max_len = max(x[0].shape[-1] for x in batch)
    
    # Pre-allocate output tensors with float16
    batch_size = len(batch)
    sample_shape = batch[0][0].shape[:-1]
    padded_data = torch.zeros(
        (batch_size, *sample_shape, max_len),
        dtype=torch.float16,  # Force float16 from start
        device='cpu'  # Keep on CPU initially
    )
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, (x, y) in enumerate(batch):
        padded_data[i, ..., :x.shape[-1]] = x.to(dtype=torch.float16)  # Convert to float16
        labels[i] = y
        
    return padded_data, labels


def create_dataloaders(train_ds, val_ds, test_ds, config):
    """Create dataloaders with weighted sampling for balanced batches"""
    # Calculate class weights for balanced sampling
    train_labels = [int(label) for _, label in train_ds]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = class_weights[train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Use weighted sampler
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate
    )
    
    # Print dataset statistics
    logger.info(f"Loader sizes: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    
    unique_labels = set()
    for i in range(len(train_ds)):
        _, label = train_ds[i]
        unique_labels.add(int(label))
    logger.info(f"Unique labels in training set: {sorted(unique_labels)}")
    
    return train_loader, val_loader, test_loader


class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, max_timepoints, transform=None):
        self.paths = paths
        self.labels = labels
        self.max_timepoints = max_timepoints
        self.transform = transform
        
        # Verify initialization
        logger.info(f"Dataset initialized with {len(paths)} samples")
        logger.info(f"Max timepoints: {max_timepoints}")
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            data = np.load(path)
            
            # Handle time dimension
            if data.shape[-1] > self.max_timepoints:
                data = data[..., :self.max_timepoints]  # Truncate
            elif data.shape[-1] < self.max_timepoints:
                pad_width = [(0, 0)] * (data.ndim - 1) + [(0, self.max_timepoints - data.shape[-1])]
                data = np.pad(data, pad_width, mode='constant')
                
            # Convert to tensor
            data = torch.from_numpy(data).float()
            
            if self.transform:
                data = self.transform(data)
                
            return data, label
            
        except Exception as e:
            logger.error(f"Error loading sample {path}: {str(e)}")
            # Return a zero tensor of the correct shape instead of None
            data = torch.zeros((data.shape[0], data.shape[1], self.max_timepoints), dtype=torch.float32)
            return data, label

