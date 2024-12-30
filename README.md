<h1 align="center">
Learned Spectrum
</h1>

<p align="center">
<em>Towards Temporal Understanding in AI through fMRI Learning Stage Classification.</em>
</p>

Full project documention can be found on: [https://teddywarner.org/Projects/LearnedSpectrum/](https://teddywarner.org/Projects/LearnedSpectrum/)

## Usage

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
python scripts/download.py
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
python scripts/download.py --force
```

### Running the Analysis

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/learnedSpectrum.ipynb`

3. Execute cells sequentially

### Experiment Tracking

1. (Optional) Set up Weights & Biases:
```bash
wandb login
```

2. View training progress:
   - Real-time metrics at [wandb.ai](https://wandb.ai)
   - Local visualizations in `visualizations/`
   - Training logs in `models/checkpoints/`

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

**Copyright © 2024 Teddy Warner**
> This work may be reproduced, modified, distributed, performed, and displayed for any purpose,
> but must acknowledge Teddy Warner. Copyright is retained and must be preserved. 
> The work is provided as is; no warranty is provided, and users accept all liability.

---
- [Watch this repo](https://github.com/Twarner491/learnedSpectrum/subscription)
- [Create issue](https://github.com/Twarner491/learnedSpectrum/issues/new)
