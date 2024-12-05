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
from enum import Enum, auto
from dataclasses import dataclass
import re
from tqdm.auto import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


class CognitiveProcess(Enum):
    ACQUISITION = auto()
    CONSOLIDATION = auto()
    TRANSFER = auto()
    REVERSAL = auto()


class TaskPhase(Enum):
    EARLY = auto()
    MIDDLE = auto()
    LATE = auto()
    MASTERY = auto()


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
    def __init__(self, config):
        self.config = config
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

    def _parallel_preprocess(self, paths: List[Path], max_workers: int = 8):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._preprocess_single, p): p for p in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc='preprocessing'):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"preprocess fail: {path} - {e}")

    def _preprocess_single(self, path: Path) -> Optional[np.ndarray]:
        cache_path = self.cache_dir / f"{path.stem}.npy"
        if cache_path.exists():
            try:
                # Load directly as float16
                data = np.load(str(cache_path), mmap_mode='r').astype(np.float16)
                return data
            except Exception:
                return None
        
        try:
            # Load and process in smaller chunks
            img = nib.load(str(path))
            data = img.get_fdata(dtype=np.float16)  # Load as float16 from start
            del img  # Free memory
            
            data = self._preprocess_volume(data)
            data = self._wavelet_transform(data)
            
            np.save(str(cache_path), data)
            return data
        except Exception as e:
            logger.error(f"preprocess fail: {path} - {e}")
            return None

    def _preprocess_volume(self, data: np.ndarray) -> np.ndarray:
        if data.ndim not in (3, 4):
            raise ValueError(f"invalid dims: {data.ndim}")
        
        if data.ndim == 3:
            data = data[..., np.newaxis]
            
        current_shape = data.shape[:3]
        target_shape = self.config.TARGET_SHAPE
        scale_factors = [t/c for t, c in zip(target_shape, current_shape)]
        data = zoom(data, scale_factors + [1], order=3)
        
        if self.config.NORMALIZE:
            data = (data - data.mean()) / (data.std() + 1e-6)
            
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
    def __init__(self, data_paths: List[Path], labels: List[int], config, nifti_loader: NiftiLoader, training: bool = True):
        valid_pairs = [(p, l) for p, l in zip(data_paths, labels)
                      if nifti_loader.load_and_preprocess(p) is not None]
        if not valid_pairs:
            raise ValueError("zero valid samples post-sanitize")
        self.data_paths, self.labels = zip(*valid_pairs)
        self.config = config
        self.nifti_loader = nifti_loader
        self.training = training
        self.augmentor = FMRIAugmentor(p=0.5 if training else 0)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        try:
            # Load data using existing interface
            data = self.nifti_loader.load_and_preprocess(self.data_paths[idx])
            if data is None:
                raise IndexError(f"corrupt idx: {idx}")
            
            # Convert to tensor efficiently
            return data, self.labels[idx]
            
        except MemoryError:
            logger.warning(f"Memory error loading {self.data_paths[idx]}")
            raise


class DatasetManager:
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.bids_manager = BIDSManager(data_config)
        self.nifti_loader = NiftiLoader(data_config)

    def _collect_dataset_samples(self, root_dir: Path) -> Tuple[List[Path], List[int]]:
        paths, labels = [], []
        for func_dir in root_dir.rglob("func"):
            for nii_file in func_dir.glob("*bold.nii.gz"):
                if self.bids_manager.validate_nifti(nii_file):
                    try:
                        dataset_id = next((ds for ds in self.data_config.DATASET_URLS.keys()
                                         if ds.split("ds")[1].lstrip("0") in str(nii_file)), None)
                        if dataset_id:
                            stage_map = self.data_config.DATASET_URLS[dataset_id]['stage_map']
                            label = int(stage_map(nii_file) * (self.config.NUM_CLASSES - 1))
                            paths.append(nii_file)
                            labels.append(label)
                            self.bids_manager.valid_files.add(nii_file)
                    except Exception as e:
                        logger.warning(f"skipped {nii_file}: {e}")
        return paths, labels

    def prepare_datasets(self):
        all_paths, all_labels = [], []
        
        # parallel dataset collection
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for dataset_id in self.data_config.DATASET_URLS.keys():
                futures.append(executor.submit(self._collect_dataset_samples, 
                                            self.bids_manager.download_dataset(dataset_id)))
            
            for future in tqdm(as_completed(futures), 
                             desc='loading datasets', 
                             total=len(futures)):
                try:
                    paths, labels = future.result()
                    all_paths.extend(paths)
                    all_labels.extend(labels)
                except Exception as e:
                    logger.error(f"dataset load fail: {e}")

        if not all_paths:
            raise ValueError("DATA_EMPTY")
            
        # parallel validation and preprocessing
        valid_paths = self.nifti_loader._parallel_validate(all_paths)
        self.nifti_loader._parallel_preprocess(valid_paths)
        
        # create datasets
        datasets = self._split_datasets(valid_paths, all_labels)
        logger.info(f"total valid: {sum(len(d) for d in datasets)}")
        
        return datasets

    def _split_datasets(self, paths, labels):
        # stratified split w/ parallel preprocessing
        valid_pairs = []
        for p, l in tqdm(zip(paths, labels), total=len(paths), desc="validating pairs"):
            data = self.nifti_loader.load_and_preprocess(p)
            if data is not None:
                valid_pairs.append((p, l))
        
        if not valid_pairs:
            raise ValueError("zero valid samples post-sanitize")
            
        paths, labels = zip(*valid_pairs)
        
        # stratified split
        label_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        splits = [], [], []  # train/val/test
        for indices in label_indices.values():
            n = len(indices)
            n_train = int(n * self.data_config.TRAIN_SPLIT)
            n_val = int(n * self.data_config.VAL_SPLIT)
            
            perm = np.random.permutation(indices)
            splits[0].extend(perm[:n_train])
            splits[1].extend(perm[n_train:n_train + n_val])
            splits[2].extend(perm[n_train + n_val:])

        return tuple(
            FMRIDataset(
                [paths[i] for i in idxs],
                [labels[i] for i in idxs],
                self.data_config,
                self.nifti_loader,
                training=(j==0)
            )
            for j, idxs in enumerate(splits)
        )
    
    
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


def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    shared_kwargs = {
        'batch_size': 4,  # Reduced from config.BATCH_SIZE
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': None,
        'drop_last': True,
        'collate_fn': pad_collate
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **shared_kwargs
    )
    
    eval_kwargs = {
        **shared_kwargs,
        'batch_size': 1,
        'shuffle': False
    }
    
    val_loader = DataLoader(val_dataset, **eval_kwargs)
    test_loader = DataLoader(test_dataset, **eval_kwargs)
    
    return train_loader, val_loader, test_loader