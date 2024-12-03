import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from functools import lru_cache

logger = logging.getLogger(__name__)

class BIDSManager:
    def __init__(self, data_config):
        self.config = data_config
        self.raw_dir = Path("../data/raw").absolute()
        self.processed_dir = Path("../data/processed").absolute()
        self.valid_files = set()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def validate_nifti(self, path: Path) -> bool:
        try:
            with open(path, 'rb') as f:
                if f.read(2) != b'\x1f\x8b':
                    return False
            img = nib.Nifti1Image.load(str(path))
            if img.header['sizeof_hdr'] != 348:
                return False
            shape = img.header.get_data_shape()
            return 3 <= len(shape) <= 4
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
                if valid:
                    return valid[0].parent.parent
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
        self.cache_dir = Path(self.config.CACHE_DIR) if self.config.USE_CACHE else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.error_count = 0
        self.max_errors = 100

    @lru_cache(maxsize=128)
    def load_and_preprocess(self, filepath: Path) -> Optional[torch.Tensor]:
        if "Classificationprobewithoutfeedback" in str(filepath):
            return None  # explicit blacklist
        try:
            if self.cache_dir:
                cache_path = self.cache_dir / f"{filepath.stem}.npy"
                if cache_path.exists():
                    return torch.from_numpy(np.load(str(cache_path)))

            img = nib.Nifti1Image.from_filename(str(filepath))
            if not img:
                raise ValueError("null image")

            data = img.get_fdata()
            if data.ndim == 3:
                data = data[..., np.newaxis]
            elif data.ndim == 4:
                data = data[..., 0][..., np.newaxis]

            if data.shape[:3] != self.config.TARGET_SHAPE:
                data = self._resize_and_pad(data, self.config.TARGET_SHAPE)

            data = torch.from_numpy(data.astype(np.float32))
            if self.config.NORMALIZE:
                data = (data - data.mean()) / (data.std() + 1e-6)

            if self.cache_dir:
                np.save(str(cache_path), data.numpy())

            return data

        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise RuntimeError(f"excessive load failures: {self.error_count}")
            logger.error(f"error loading {filepath}: {e}")
            return torch.zeros((*self.config.TARGET_SHAPE, 1), dtype=torch.float32)

    def _resize_and_pad(self, data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        current_shape = data.shape[:3]
        resized = np.zeros((*target_shape, data.shape[-1]), dtype=data.dtype)
        
        min_h = min(current_shape[0], target_shape[0])
        min_w = min(current_shape[1], target_shape[1])
        min_d = min(current_shape[2], target_shape[2])
        
        resized[:min_h, :min_w, :min_d, :] = data[:min_h, :min_w, :min_d, :]
        return resized

class FMRIDataset(Dataset):
    def __init__(self, data_paths, labels, config, nifti_loader: NiftiLoader):
        # pre-sanitize on init
        valid_samples = []
        valid_labels = []
        for path, label in zip(data_paths, labels):
            data = nifti_loader.load_and_preprocess(path)
            if data is not None:  # explicit None check
                valid_samples.append(path)
                valid_labels.append(label)
        
        if not valid_samples:
            raise ValueError("no valid samples after sanitization")
            
        self.data_paths = valid_samples
        self.labels = valid_labels
        self.config = config
        self.nifti_loader = nifti_loader

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = self.nifti_loader.load_and_preprocess(self.data_paths[idx])
        if data is None:  # paranoid double-check
            raise IndexError(f"corrupt data at {idx}")
        return data, self.labels[idx]
    
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
        if not paths:
            raise ValueError(f"no valid volumes in {root_dir}")
        return paths, labels

    def _split_datasets(self, all_paths: List[Path], all_labels: List[int]):
        n = len(all_paths)
        indices = np.random.permutation(n)
        train_idx = indices[:int(n * self.data_config.TRAIN_SPLIT)]
        val_idx = indices[int(n * self.data_config.TRAIN_SPLIT):
                         int(n * (self.data_config.TRAIN_SPLIT + self.data_config.VAL_SPLIT))]
        test_idx = indices[int(n * (self.data_config.TRAIN_SPLIT + self.data_config.VAL_SPLIT)):]
        
        train_dataset = FMRIDataset(
            [all_paths[i] for i in train_idx],
            [all_labels[i] for i in train_idx],
            self.data_config,
            self.nifti_loader
        )
        val_dataset = FMRIDataset(
            [all_paths[i] for i in val_idx],
            [all_labels[i] for i in val_idx],
            self.data_config,
            self.nifti_loader
        )
        test_dataset = FMRIDataset(
            [all_paths[i] for i in test_idx],
            [all_labels[i] for i in test_idx],
            self.data_config,
            self.nifti_loader
        )
        
        logger.info(f"splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def prepare_datasets(self):
        all_paths, all_labels = [], []
        for dataset_id in self.data_config.DATASET_URLS.keys():
            try:
                root = self.bids_manager.download_dataset(dataset_id)
                paths, labels = self._collect_dataset_samples(root)
                all_paths.extend(paths)
                all_labels.extend(labels)
            except Exception as e:
                logger.error(f"dataset {dataset_id} failed: {e}")
                continue
        if not all_paths:
            raise ValueError("no valid datasets found")
        return self._split_datasets(all_paths, all_labels)

def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    def collate_fn(batch):
        # additional sanitizer at batch level
        valid_batch = [(x, y) for x, y in batch if x is not None]
        if not valid_batch:
            raise RuntimeError("entire batch was corrupt")
        return torch.utils.data.dataloader.default_collate(valid_batch)

    return (
        DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                  num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                  collate_fn=collate_fn),
        DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                  num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                  collate_fn=collate_fn),
        DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                  num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                  collate_fn=collate_fn)
    )