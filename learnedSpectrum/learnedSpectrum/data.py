"""
Data management classes and functions for LearnedSpectrum
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import urllib.request
import zipfile
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pywt
from scipy.ndimage import zoom
from nilearn import image
from functools import lru_cache

logger = logging.getLogger(__name__)

class BIDSManager:
    """Handles dataset organization according to BIDS format"""
    
    def __init__(self, data_config):
        self.config = data_config
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str) -> Path:
        """Downloads dataset if not already present"""
        target_dir = self.raw_dir / dataset_id
        if target_dir.exists():
            logger.info(f"Dataset {dataset_id} already downloaded")
            return target_dir
            
        url = self.config.DATASET_URLS[dataset_id]
        zip_path = self.raw_dir / f"{dataset_id}.zip"
        
        logger.info(f"Downloading {dataset_id}...")
        urllib.request.urlretrieve(url, zip_path)
        
        logger.info(f"Extracting {dataset_id}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        zip_path.unlink()  # Remove zip file after extraction
        return target_dir

class FMRIDataset(Dataset):
    """Dataset class for fMRI volumes"""
    
    def __init__(self, 
                 data_paths: List[Path],
                 labels: List[int],
                 config,
                 transform=None,
                 cache_dir: Optional[Path] = None):
        self.data_paths = data_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=128)
    def _load_and_preprocess(self, idx: int) -> np.ndarray:
        """Load and preprocess a single volume with caching"""
        path = self.data_paths[idx]
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{path.stem}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load and preprocess
        img = nib.load(path)
        data = img.get_fdata()
        
        # Resize to target shape
        if data.shape != self.config.TARGET_SHAPE:
            zoom_factors = [t / c for t, c in zip(self.config.TARGET_SHAPE, data.shape)]
            data = zoom(data, zoom_factors)
        
        # Apply wavelet transform if configured
        if self.config.USE_WAVELET:
            coeffs = pywt.wavedec3(
                data, 
                wavelet=self.config.WAVELET_NAME,
                level=self.config.DECOMPOSITION_LEVEL
            )
            data = np.stack([c[0] for c in coeffs], axis=0)
        
        # Normalize
        if self.config.NORMALIZE:
            data = (data - data.mean()) / (data.std() + 1e-6)
        
        # Cache result
        if self.cache_dir:
            np.save(cache_path, data)
            
        return data

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self._load_and_preprocess(idx)
        data = torch.from_numpy(data).float()
        
        if self.transform:
            data = self.transform(data)
            
        return data, self.labels[idx]

class DatasetManager:
    """Manages dataset loading, splitting, and dataloader creation"""
    
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.bids_manager = BIDSManager(data_config)

    def prepare_datasets(self) -> Tuple[FMRIDataset, FMRIDataset, FMRIDataset]:
        """Prepare train, validation and test datasets"""
        # Download and organize all datasets
        all_paths = []
        all_labels = []
        
        for dataset_id in self.data_config.DATASET_URLS.keys():
            dataset_dir = self.bids_manager.download_dataset(dataset_id)
            paths, labels = self._collect_dataset_samples(dataset_dir)
            all_paths.extend(paths)
            all_labels.extend(labels)
        
        # Split data
        indices = np.random.permutation(len(all_paths))
        train_size = int(len(indices) * self.data_config.TRAIN_SPLIT)
        val_size = int(len(indices) * self.data_config.VAL_SPLIT)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create datasets
        cache_dir = Path(self.data_config.CACHE_DIR) if self.data_config.USE_CACHE else None
        
        train_dataset = FMRIDataset(
            [all_paths[i] for i in train_idx],
            [all_labels[i] for i in train_idx],
            self.data_config,
            cache_dir=cache_dir
        )
        
        val_dataset = FMRIDataset(
            [all_paths[i] for i in val_idx],
            [all_labels[i] for i in val_idx],
            self.data_config,
            cache_dir=cache_dir
        )
        
        test_dataset = FMRIDataset(
            [all_paths[i] for i in test_idx],
            [all_labels[i] for i in test_idx],
            self.data_config,
            cache_dir=cache_dir
        )
        
        return train_dataset, val_dataset, test_dataset

    def _collect_dataset_samples(self, dataset_dir: Path) -> Tuple[List[Path], List[int]]:
        """Collect all valid samples and their labels from a dataset"""
        paths = []
        labels = []
        
        # Implementation depends on specific BIDS structure
        # This is a simplified version
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('_bold.nii.gz'):
                    paths.append(Path(root) / file)
                    # Extract label from filename or metadata
                    # This is placeholder logic
                    label = len(labels) % self.config.NUM_CLASSES
                    labels.append(label)
                    
        return paths, labels

def create_dataloaders(train_dataset: FMRIDataset,
                      val_dataset: FMRIDataset,
                      test_dataset: FMRIDataset,
                      config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation and test dataloaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    return train_loader, val_loader, test_loader
