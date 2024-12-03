"""
Data preprocessing pipeline for fMRI data
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import nibabel as nib
import numpy as np
from nilearn import image
import pywt
from scipy.ndimage import zoom
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config.ROOT) / "data/processed"
        self.cache_dir = Path(config.ROOT) / "data/cleaned"
        self.raw_dir = Path(config.ROOT) / "data/raw"
        
        for dir_path in [self.processed_dir, self.cache_dir, self.raw_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_volume(self, 
                      img_path: Path,
                      target_shape: Tuple[int, int, int] = (64, 64, 30)) -> np.ndarray:
        """Process a single fMRI volume"""
        # Load the NIfTI file
        img = nib.load(str(img_path))
        
        # Standardize orientation to RAS+
        img = nib.as_closest_canonical(img)
        
        # Get data array
        data = img.get_fdata()
        
        # Resize to target shape
        if data.shape != target_shape:
            zoom_factors = [t / c for t, c in zip(target_shape, data.shape)]
            data = zoom(data, zoom_factors, order=3)  # order=3 for cubic interpolation
        
        # Apply wavelet transform if configured
        if self.config.USE_WAVELET:
            coeffs = pywt.wavedec3(
                data, 
                wavelet=self.config.WAVELET_NAME,
                level=self.config.DECOMPOSITION_LEVEL
            )
            # Stack coefficients
            data = np.stack([c[0] for c in coeffs], axis=0)
        
        # Normalize
        if self.config.NORMALIZE:
            data = (data - data.mean()) / (data.std() + 1e-6)
            
        return data

    def process_subject(self, 
                       subject_dir: Path,
                       task_name: str = "learning") -> List[Tuple[np.ndarray, int]]:
        """Process all task runs for a single subject"""
        processed_data = []
        
        # Find all task runs
        func_dir = subject_dir / "func"
        if not func_dir.exists():
            logger.warning(f"No func directory found for subject {subject_dir.name}")
            return processed_data
        
        # Process each run
        for run_file in func_dir.glob(f"*task-{task_name}*_bold.nii.gz"):
            try:
                # Extract run number and learning stage from filename
                run_info = run_file.stem.split("_")
                run_num = next(part for part in run_info if part.startswith("run-"))
                run_num = int(run_num.split("-")[1])
                
                # Determine learning stage (this logic should match your specific dataset)
                stage = self._determine_learning_stage(run_file)
                
                # Process volume
                processed_volume = self.process_volume(run_file)
                
                processed_data.append((processed_volume, stage))
                
            except Exception as e:
                logger.error(f"Error processing {run_file}: {str(e)}")
                continue
                
        return processed_data

    def _determine_learning_stage(self, run_file: Path) -> int:
        """Determine learning stage from run file and metadata"""
        # This is a placeholder - implement according to your dataset structure
        # You might need to read associated JSON files or other metadata
        # For now, we'll use a simple mapping based on run number
        run_num = int(next(part for part in run_file.stem.split("_") 
                         if part.startswith("run-")).split("-")[1])
        
        if run_num <= 2:
            return 0  # Early learning
        elif run_num <= 4:
            return 1  # Middle learning
        elif run_num <= 6:
            return 2  # Late learning
        else:
            return 3  # Mastery

    def process_dataset(self, 
                       dataset_dir: Path,
                       cache: bool = True) -> List[Tuple[np.ndarray, int]]:
        """Process entire dataset"""
        all_data = []
        
        # Find all subjects
        subject_dirs = [d for d in dataset_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("sub-")]
        
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            # Check cache first if enabled
            if cache:
                cache_file = self.processed_dir / f"{subject_dir.name}.npz"
                if cache_file.exists():
                    with np.load(cache_file) as data:
                        all_data.extend([(data['volumes'][i], data['labels'][i]) 
                                       for i in range(len(data['labels']))])
                    continue
            
            # Process subject
            subject_data = self.process_subject(subject_dir)
            
            # Cache results if enabled
            if cache and subject_data:
                volumes, labels = zip(*subject_data)
                np.savez(cache_file, 
                        volumes=np.array(volumes),
                        labels=np.array(labels))
            
            all_data.extend(subject_data)
        
        return all_data

def main():
    """Main function for standalone processing"""
    import argparse
    from ..learnedSpectrum.config import DataConfig
    
    parser = argparse.ArgumentParser(description="Process fMRI datasets")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing BIDS dataset")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching of processed data")
    args = parser.parse_args()
    
    config = DataConfig()
    processor = DataProcessor(config)
    
    input_dir = Path(args.input_dir)
    processor.process_dataset(input_dir, cache=not args.no_cache)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 