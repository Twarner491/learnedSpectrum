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
import shutil
import gzip


logger = logging.getLogger(__name__)

class BIDSManager:
    def __init__(self, data_config):
        self.config = data_config
        self.raw_dir = Path("../data/raw").absolute()
        self.processed_dir = Path("../data/processed").absolute()
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _find_dataset_root(self, dataset_id: str) -> Path:
        # handle openfmri's *delightful* inconsistency w/ naming
        patterns = [
            f"{dataset_id}",  # base
            f"{dataset_id}_R*.0.*",  # versioned
            f"ds{dataset_id.split('ds')[1]}_R*.0.*",  # alt format
            "*/*bold.nii.gz"  # desperate recursion
        ]
        
        for pattern in patterns:
            candidates = list(self.raw_dir.rglob(pattern))
            if candidates:
                # take deepest match w/ func dir
                valid = [p for p in candidates if "func" in str(p.parent)]
                if valid:
                    return valid[0].parent.parent  # climb back to subject
                return candidates[0]  # fallback to first
                
        raise FileNotFoundError(f"dataset {dataset_id} not found in expected locations")

    def download_dataset(self, dataset_id: str) -> Path:
        """handle bids dataset location w/ version chaos"""
        try:
            root = self._find_dataset_root(dataset_id)
            logging.info(f"found dataset root: {root}")
            return root
        except Exception as e:
            logging.error(f"failed to locate {dataset_id}: {e}")
            raise

class FMRIDataset(Dataset):
    def __init__(self, data_paths, labels, config, transform=None, cache_dir=None):
        self.data_paths = data_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_preprocess(self, idx):
        path = Path(self.data_paths[idx])
        
        if self.cache_dir:
            cache_path = self.cache_dir / f"{path.stem}.npy"
            if cache_path.exists():
                try:
                    return np.load(str(cache_path))
                except:
                    pass
        
        try:
            # robust loading w/ error handling
            img = nib.load(str(path))
            data = img.get_fdata()
            
            if data.ndim > 3:
                data = np.squeeze(data)  # remove singleton dims
            
            if data.shape != self.config.TARGET_SHAPE:
                zoom_factors = [t / c for t, c in zip(self.config.TARGET_SHAPE, data.shape)]
                data = zoom(data, zoom_factors, order=1)  # linear interp
            
            if self.config.NORMALIZE:
                data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
            if self.cache_dir:
                np.save(str(cache_path), data)
            
            return data
            
        except Exception as e:
            print(f"error loading {path}: {str(e)}")
            # fallback: return zero array of correct shape
            return np.zeros(self.config.TARGET_SHAPE)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = self._load_and_preprocess(idx)
        data = torch.from_numpy(data).float()
        
        if self.transform:
            data = self.transform(data)
        
        return data, self.labels[idx]

class DatasetManager:
    def __init__(self, config, data_config):
        self.config = config  # main config
        self.data_config = data_config  # crucial for urls/mapping
        self.bids_manager = BIDSManager(data_config)

    def _collect_dataset_samples(self, root_dir: Path) -> Tuple[List[Path], List[int]]:
        paths, labels = [], []
        try:
            for func_dir in root_dir.rglob("func"):
                for nii_file in func_dir.glob("*bold.nii.gz"):
                    try:
                        img = nib.load(str(nii_file))
                        _ = img.header
                        dataset_id = next(
                            (ds for ds in self.data_config.DATASET_URLS.keys()  # <-- fixed config access
                             if ds.split("ds")[1].lstrip("0") in str(nii_file)),  # handle ds002 vs ds000002
                            None
                        )
                        if dataset_id:
                            stage_map = self.data_config.DATASET_URLS[dataset_id]['stage_map']
                            label = int(stage_map(nii_file) * (self.config.NUM_CLASSES - 1))
                            paths.append(nii_file)
                            labels.append(label)
                            logging.debug(f"added {nii_file} (label={label})")
                    except Exception as e:
                        logging.warning(f"skipped {nii_file}: {e}")
        except Exception as e:
            logging.error(f"scan failed {root_dir}: {e}")
            
        if not paths:
            raise ValueError(f"no valid volumes in {root_dir}")
            
        return paths, labels
    
    def _split_datasets(self, all_paths: List[Path], all_labels: List[int]):
        """hochbergian train/val/test split"""
        n = len(all_paths)
        indices = np.random.permutation(n)
        
        train_idx = indices[:int(n * self.data_config.TRAIN_SPLIT)]
        val_idx = indices[int(n * self.data_config.TRAIN_SPLIT):
                        int(n * (self.data_config.TRAIN_SPLIT + self.data_config.VAL_SPLIT))]
        test_idx = indices[int(n * (self.data_config.TRAIN_SPLIT + self.data_config.VAL_SPLIT)):]
        
        # instantiate w/ correct config
        cache_dir = Path(self.data_config.CACHE_DIR) if self.data_config.USE_CACHE else None
        
        splits = (
            [FMRIDataset([all_paths[i] for i in idx], 
                        [all_labels[i] for i in idx],
                        self.data_config,
                        cache_dir=cache_dir)
            for idx in (train_idx, val_idx, test_idx)]
        )
        
        logging.info(f"splits: train={len(splits[0])}, val={len(splits[1])}, test={len(splits[2])}")
        return splits

    def prepare_datasets(self):
        all_paths, all_labels = [], []
        
        for dataset_id in self.data_config.DATASET_URLS.keys():
            try:
                root = self.bids_manager.download_dataset(dataset_id)
                paths, labels = self._collect_dataset_samples(root)
                all_paths.extend(paths)
                all_labels.extend(labels)
            except Exception as e:
                logging.error(f"dataset {dataset_id} failed: {e}")
                continue
                
        if not all_paths:
            raise ValueError("no valid datasets found")
            
        return self._split_datasets(all_paths, all_labels)

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
