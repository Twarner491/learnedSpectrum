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
        'url': 'https://s3.amazonaws.com/openneuro/ds000002/ds000002_R2.0.5/compressed/ds000002_R2.0.5_raw.zip',
        'description': 'Classification Learning',
        'tr': 2.0,
        'stage_map': lambda f: 0.25 if 'run-2' in str(f) else 0.0  # prob learning
    },
    'ds000011': {
        'url': 'https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/compressed/ds000011_R2.0.1_raw.zip',
        'description': 'Mixed-gambles Task',
        'tr': 1.5,
        'stage_map': lambda f: 0.5 if 'run-2' in str(f) else 0.25  # det learning
    },
    'ds000017': {
        'url': 'https://s3.amazonaws.com/openneuro/ds000017/ds000017_R2.0.1/compressed/ds000017_R2.0.1.zip',
        'description': 'Classification Learning and Reversal',
        'tr': 2.5,
        'stage_map': lambda f: 0.75 if 'run-2' in str(f) else 0.5  # reversal
    },
    'ds000052': {
        'url': 'https://s3.amazonaws.com/openneuro/ds000052/ds000052_R2.0.0/compressed/ds052_R2.0.0_01-14.tgz',
        'description': 'Classification Learning and Stop-signal',
        'tr': 2.0,
        'stage_map': lambda f: 1.0 if ('reversal' in str(f) and 'run-2' in str(f)) else
                              0.67 if ('reversal' in str(f)) else
                              0.33 if 'run-2' in str(f) else 0.0  # full spectrum
    }
}


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
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
    Extract zip or tar.gz/tgz archive
    """
    logger.info(f"Extracting {archive_path} to {extract_path}")
    
    # Create extraction directory if it doesn't exist
    extract_path.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.suffix in ['.gz', '.tgz'] or archive_path.name.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted path traversal in tar file")
                
                tar.extractall(path, members, numeric_owner=numeric_owner)
            
            safe_extract(tar_ref, str(extract_path))
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


def get_file_extension(url: str) -> str:
    """
    Get the file extension from the URL
    """
    if url.endswith('.zip'):
        return '.zip'
    elif url.endswith('.tgz'):
        return '.tgz'
    elif url.endswith('.tar.gz'):
        return '.tar.gz'
    else:
        raise ValueError(f"Unsupported file format in URL: {url}")
    

def download_datasets(force_download: bool = False) -> None:
    """
    Download and extract all required datasets
    """
    raw_dir = setup_data_directories()
    success = True

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
            # Get correct file extension from URL
            file_ext = get_file_extension(info['url'])
            # Download archive with correct extension
            archive_path = temp_dir / f"{dataset_id}{file_ext}"
            download_file(info['url'], archive_path)
            
            # Extract to raw directory
            extract_archive(archive_path, dataset_dir)
            
            logger.info(f"Successfully downloaded and extracted {dataset_id}")
            
        except Exception as e:
            success = False
            logger.error(f"Error processing {dataset_id}: {str(e)}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
        
        finally:
            # Cleanup temporary files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    if success:
        logger.info("All datasets downloaded successfully")
    else:
        logger.error("Some datasets failed to download. Please check the errors above.")
        sys.exit(1)
        

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