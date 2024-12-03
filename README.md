<h1 align="center">
Learned Spectrum
</h1>

<p align="center">
<em>fMRI Learning Stage Classification with Vision Transformers</em>
</p>

<div align="center">

![build](https://github.com/buttons/github-buttons/workflows/build/badge.svg)
[![hf](https://img.shields.io/badge/spaces-blue?style=flat&logo=huggingface&logoColor=darkgrey&label=Hugging%20Face&labelColor=grey)](https://huggingface.co/spaces/twarner/learnedSpectrum)

</div>


## Introduction

I came into this project intent on focusing on the "medical-tangent" space with a specific interest in brain imaging, and wound up studying/utilizing fMRI data. My naivety with fMRI sure got the best of me here, as boy was the data a pain in the ass to wrap my head around, yet I had an absolute blast doing so.

As a means of establishing a theoretical baseline regarding fMRI and neuroimaging, I interviewed Talha Rafique, a researcher at USC Institute for Technology and Medical Systems's Khan Lab. While doing so, we reached the conclusion that attempting to extract the different stages of learning from fMRI data is less than ideal, due to the somewhat ambiguous nature of fMRI. At the risk of significant oversimplification, fMRI captures where the oxygenated blood is flowing in your brain (through detecting iron concentration with big magnets). While doing so may tell us what regions of the brain are currently "active", results from fMRI may be slightly misleading, as there is oxygenated blood everywhere in your brain, and its concentration doesn't vary vastly enough to truly distill down what region of the brain is truly most "active".

Talha suggested that I povit to EEG data, as doing so would allow for much higher confidence in brain region activation during a task (such as learning/recall). Yet for one reason or another I decided to stick with fMRI to see if a model would have more luck at pulling significance from the data than humans have had historically (also because I wanted to work with Vision Transformers and EEG data is tabular). Thus my IDSN 544: Transformative AI in Society final project: Learned Spectrum, an attempt at Learning Stage Classification from fMRI data with Vision Transformers.

**Copyright © 2024 Teddy Warner**
> This work may be reproduced, modified, distributed, performed, and displayed for any purpose,
> but must acknowledge Teddy Warner. Copyright is retained and must be preserved. 
> The work is provided as is; no warranty is provided, and users accept all liability.

## Abstract

This project implements a Vision Transformer (ViT) based approach for classifying learning stages from functional Magnetic Resonance Imaging (fMRI) data. Using four classification learning datasets from OpenFMRI, we develop a deep learning model to identify distinct phases of learning from brain activation patterns. The model processes 3D fMRI volumes as sequences of 2D slices, leveraging the transformer architecture's ability to capture long-range dependencies in spatial and temporal dimensions.

The implementation utilizes robust preprocessing pipelines for fMRI data normalization, including spatial resizing, intensity normalization, and temporal resolution standardization. The model is evaluated on its ability to distinguish between different learning stages (probabilistic, deterministic, and mixed conditions) across multiple experimental paradigms.

This work explores the potential of modern deep learning architectures in understanding neural correlates of learning, offering a novel approach to analyzing complex neuroimaging data that has historically been challenging to interpret through traditional methods.

## Usage

**Google Colab** 
> 1. Open the notebook in Google Colab by clicking [this link](https://colab.research.google.com/drive/1sJd9nWyGymv6NzhaKcB-Y0KHW_PmF9RN?usp=sharing)
> 2. Execute the notebook chronologically
> 3. Make sure to enable A100 GPU runtime for optimal performance

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Twarner491/learnedSpectrum.git
cd learnedSpectrum
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package in editable mode with all dependencies:
```bash
pip install -e .
```

### Data Preparation

1. Create necessary directories and download datasets (automated):
```bash
python scripts/download_datasets.py
```

This script will:
- Create required directory structure
- Download OpenFMRI datasets:
  - [ds000002](https://openneuro.org/datasets/ds000002) - Classification Learning
  - [ds000011](https://openneuro.org/datasets/ds000011) - Mixed-gambles Task
  - [ds000017](https://openneuro.org/datasets/ds000017) - Classification Learning and Reversal
  - [ds000052](https://openneuro.org/datasets/ds000052) - Classification Learning and Stop-signal
- Extract datasets to the correct locations
- Clean up temporary files

The resulting directory structure will be:
```
data/
├── raw/
│   ├── ds000002/
│   ├── ds000011/
│   ├── ds000017/
│   └── ds000052/
├── processed/
└── cleaned/
```

Options:
```bash
# Force redownload of datasets (if needed)
python scripts/download_datasets.py --force
```

### Running the Analysis

#### Option 1: Using Jupyter Notebook (Recommended)

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/learnedSpectrum.ipynb`

3. The notebook is organized into clear sections:
   - Setup and Imports
   - Configuration
   - Data Preparation
   - Model Training
   - Evaluation
   - Results Visualization

4. Execute cells sequentially to:
   - Process the fMRI data
   - Train the Vision Transformer model
   - Visualize results and attention maps
   - Generate performance metrics

#### Option 2: Command Line Interface

1. Process the datasets:
```bash
python scripts/data_processor.py --input_dir data/raw
```

2. Train the model:
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --output_dir models/checkpoints \
    --wandb  # Optional: Enable Weights & Biases logging
```

3. Evaluate and visualize results:
```bash
python scripts/visualization.py \
    --model_path models/checkpoints/best_model.pth \
    --output_dir visualizations
```

### Experiment Tracking

1. (Optional) Set up Weights & Biases:
```bash
wandb login
```

2. View training progress:
   - Real-time metrics at [wandb.ai](https://wandb.ai)
   - Local visualizations in `visualizations/`
   - Training logs in `models/checkpoints/`

### Output Files

After running the analysis, you'll find:
- Processed data: `data/processed/`
- Model checkpoints: `models/checkpoints/`
- Visualizations:
  - Brain slices: `visualizations/brain_slices/`
  - Attention maps: `visualizations/attention_maps/`
  - Performance plots: `visualizations/metrics/`
  - ROC curves: `visualizations/roc_curves.png`
  - Confusion matrix: `visualizations/confusion_matrix.png`

### Troubleshooting

1. CUDA/GPU Issues:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

2. Memory Issues:
   - Reduce `BATCH_SIZE` in config
   - Enable gradient accumulation
   - Use mixed precision training

3. Data Loading Issues:
   - Verify dataset structure
   - Check preprocessing logs in `data/processed/logs/`
   - Ensure sufficient disk space

## Acknowledgements 
A thank you to Talha Rafique and the USC Institute for Technology and Medical Systems's Khan Lab for providing guidance on this project.

---
- [Watch this repo](https://github.com/Twarner491/learnedSpectrum/subscription)
- [Create issue](https://github.com/Twarner491/learnedSpectrum/issues/new)
