"""
Dataset downloader for LearnedSpectrum
Downloads and extracts required OpenFMRI datasets
"""

import os
import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset information
DATASETS = {
    'ds000002': {
        'url': 'https://openneuro.org/crn/datasets/ds000002/snapshots/00001/files',
        'description': 'Classification Learning'
    },
    'ds000011': {
        'url': 'https://openneuro.org/crn/datasets/ds000011/snapshots/00001/files',
        'description': 'Mixed-gambles Task'
    },
    'ds000017': {
        'url': 'https://openneuro.org/crn/datasets/ds000017/snapshots/00001/files',
        'description': 'Classification Learning and Reversal'
    },
    'ds000052': {
        'url': 'https://openneuro.org/crn/datasets/ds000052/snapshots/00001/files',
        'description': 'Classification Learning and Stop-signal'
    }
}

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def extract_archive(archive_path: Path, extract_path: Path) -> None:
    """
    Extract zip or tar.gz archive
    """
    logger.info(f"Extracting {archive_path} to {extract_path}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.suffix == '.gz' and archive_path.suffixes[-2:] == ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

def setup_data_directories() -> Path:
    """
    Create necessary data directories
    """
    data_dir = project_root / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    cleaned_dir = data_dir / 'cleaned'

    for directory in [raw_dir, processed_dir, cleaned_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return raw_dir

def download_datasets(force_download: bool = False) -> None:
    """
    Download and extract all required datasets
    """
    raw_dir = setup_data_directories()

    for dataset_id, info in DATASETS.items():
        dataset_dir = raw_dir / dataset_id
        
        if dataset_dir.exists() and not force_download:
            logger.info(f"Dataset {dataset_id} already exists. Skipping...")
            continue

        logger.info(f"Processing {dataset_id} - {info['description']}")
        
        # Create temporary directory for downloads
        temp_dir = raw_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download archive
            archive_path = temp_dir / f"{dataset_id}.zip"
            download_file(info['url'], archive_path)
            
            # Extract to raw directory
            extract_archive(archive_path, dataset_dir)
            
            logger.info(f"Successfully downloaded and extracted {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_id}: {str(e)}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
        
        finally:
            # Cleanup temporary files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download OpenFMRI datasets for LearnedSpectrum')
    parser.add_argument('--force', action='store_true', help='Force download even if datasets exist')
    args = parser.parse_args()

    try:
        download_datasets(force_download=args.force)
        logger.info("Dataset download completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 