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

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- nibabel
- nilearn
- scikit-learn
- transformers
- wandb (for experiment tracking)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/learnedSpectrum.git
cd learnedSpectrum
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

### Running the Code

#### Option 1: Using Google Colab (Recommended)
1. Open the notebook in Google Colab by clicking [this link](https://colab.research.google.com/github/yourusername/learnedSpectrum/blob/main/notebooks/learnedSpectrum.ipynb)
2. Follow the step-by-step instructions in the notebook
3. Make sure to enable GPU runtime for optimal performance

#### Option 2: Local Execution
1. Download the required datasets from OpenFMRI:
   - ds000002
   - ds000011
   - ds000017
   - ds000052

2. Place the downloaded datasets in the `data/` directory

3. Run the preprocessing pipeline:
```bash
python -m learnedSpectrum.preprocess --data_dir data/ --output_dir processed/
```

4. Train the model:
```bash
python -m learnedSpectrum.train --data_dir processed/ --output_dir models/
```

5. Evaluate the model:
```bash
python -m learnedSpectrum.evaluate --model_path models/best_model.pth --data_dir processed/
```


## Acknowledgements 
A thank you to Talha Rafique and the USC Institute for Technology and Medical Systems's Khan Lab for providing guidance on this project.

---
- [Watch this repo](https://github.com/Twarner491/learnedSpectrum/subscription)
- [Create issue](https://github.com/Twarner491/learnedSpectrum/issues/new)
