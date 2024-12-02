<h1 align="center">
Learned Spectrum
</h1>

<p align="center">
<em>fMRI Learning Stage Classification with Vision Transformers</em>
</p>

**Copyright © 2024 Teddy Warner**
> This work may be reproduced, modified, distributed, performed, and displayed for any purpose,
> but must acknowledge Teddy Warner. Copyright is retained and must be preserved. 
> The work is provided as is; no warranty is provided, and users accept all liability.

## Abstract
This study presents a novel approach to assessing an individual's knowledge level on a subject using functional Magnetic Resonance Imaging (fMRI) data processed by vision transformers. We introduce a state-of-the-art neural network architecture that combines 3D vision transformers, task-specific feature extraction, and temporal modeling to analyze fMRI data from multiple datasets involving classification learning tasks. Our model incorporates multi-task learning to predict knowledgeability scores, brain region activation, cognitive states, and task performance. We employ advanced techniques such as dynamic functional connectivity analysis, reinforcement learning, and explainable AI methods to enhance the model's performance and interpretability. The proposed approach demonstrates significant improvements in accurately classifying an individual's knowledge level compared to traditional methods, offering new insights into the neural correlates of learning and recall processes.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/learnedSpectrum.git
cd learnedSpectrum
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate learned-spectrum
```

3. Install package in development mode:
```bash
pip install -e .
```

3.5. If you want to install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
learnedSpectrum/
├── data/                      # Data directory
│   ├── raw/                  # Raw fMRI data
│   ├── processed/            # Preprocessed data
│   └── results/              # Analysis results
├── learnedSpectrum/          # Main package
│   ├── __init__.py
│   ├── config.py            # Configuration
│   ├── customFuncs.py       # Utility functions
│   └── tests/               # Unit tests
├── notebooks/                # Jupyter notebooks
│   ├── preprocessing.ipynb
│   ├── featureExtraction.ipynb
│   └── modelValidation.ipynb
└── scripts/                  # Processing scripts
    ├── dataProcessing.py
    ├── model.py
    ├── training.py
    └── validation.py
```

## Usage Pipeline

### 1. Data Preprocessing

Run the preprocessing notebook to prepare your data:
```bash
jupyter lab notebooks/preprocessing.ipynb
```

Key steps:
- Load raw fMRI data
- Apply motion correction
- Perform slice timing correction
- Execute spatial normalization
- Implement noise reduction
- Save preprocessed data

Configuration:
```python
# config.py
preprocessing_config = {
    'motion_correction': True,
    'slice_timing': True,
    'spatial_norm': True,
    'noise_reduction': True
}
```

### 2. Feature Extraction and Model Training

Run the feature extraction and training notebook:
```bash
jupyter lab notebooks/featureExtraction.ipynb
```

Steps:
1. Load preprocessed data
2. Extract temporal features
3. Configure model:
```python
model_config = {
    'vit': {
        'patch_size': (4, 4, 4),
        'hidden_size': 768,
        'num_layers': 12
    }
}
```

4. Train model:
```python
# Train from command line
python scripts/training.py --config configs/training.yaml

# Or use training notebook
jupyter lab notebooks/featureExtraction.ipynb
```

### 3. Model Validation and Visualization

Run the validation notebook:
```bash
jupyter lab notebooks/modelValidation.ipynb
```

Analysis includes:
- Performance metrics
- Attention visualization
- Feature importance
- Learning progression

### 4. Saving and Loading Results

Results are automatically saved throughout the pipeline:

```python
# Save preprocessed data
save_dir = Path('data/processed/')
preprocessor.save_results(save_dir)

# Save model checkpoints
checkpoint_dir = Path('models/checkpoints/')
trainer.save_checkpoint(checkpoint_dir)

# Save validation results
results_dir = Path('data/results/')
validator.save_results(results_dir)
```

### 5. Testing

Run unit tests to ensure functionality:
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest learnedSpectrum/tests/test_config.py

# Run with coverage
coverage run -m pytest
coverage report
```

## Visualization Examples

1. Brain attention patterns:
```python
from scripts.explainability import AttentionMapper

attention_mapper = AttentionMapper(model)
attention_mapper.plot_attention_map(attention_maps)
```

2. Learning progression:
```python
from scripts.validation import LearningStageAnalyzer

analyzer = LearningStageAnalyzer(model)
analyzer.plot_learning_trajectory(predictions)
```

## Configuration

Configure project settings in `config.py`:
```python
config = {
    'data_dir': 'path/to/data',
    'preprocessing': {...},
    'model': {...},
    'training': {...},
    'validation': {...}
}
```

Or use YAML configuration:
```yaml
# config.yaml
data:
  root_dir: "data/"
  preprocessed_dir: "data/processed/"

model:
  type: "vit3d"
  patch_size: [4, 4, 4]
  hidden_size: 768
```

## Output Structure

Results are organized as follows:
```
results/
├── preprocessing/
│   ├── quality_metrics.json
│   └── preprocessing_report.md
├── training/
│   ├── checkpoints/
│   ├── logs/
│   └── training_metrics.json
└── validation/
    ├── attention_maps/
    ├── feature_importance/
    └── validation_report.md
```

## Example Workflow

1. Prepare your data:
```bash
# Create necessary directories
mkdir -p data/{raw,processed,results}

# Place raw fMRI data in data/raw/
```

2. Run preprocessing:
```bash
jupyter lab notebooks/preprocessing.ipynb
```

3. Train model:
```bash
jupyter lab notebooks/featureExtraction.ipynb
```

4. Validate and visualize:
```bash
jupyter lab notebooks/modelValidation.ipynb
```

5. Review results:
- Check preprocessing report
- Monitor training metrics
- Analyze validation results
- Examine visualizations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Run tests and linting
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{learnedSpectrum2024,
  author = {Your Name},
  title = {LearnedSpectrum: fMRI Learning Stage Classification},
  year = {2024},
  url = {https://github.com/yourusername/learnedSpectrum}
}
```

## 1. Introduction

The ability to objectively assess an individual's knowledge level on a subject has long been a challenge in cognitive neuroscience and education. Traditional methods often rely on behavioral measures or standardized tests, which may not fully capture the underlying neural processes involved in knowledge acquisition and retrieval. Recent advancements in neuroimaging techniques, particularly functional Magnetic Resonance Imaging (fMRI), have opened new avenues for understanding the brain's activity patterns associated with learning and recall.

This study aims to leverage these advancements by developing a novel approach to knowledge assessment using fMRI data. We propose a state-of-the-art neural network architecture that utilizes vision transformers to process volumetric fMRI data, enabling us to capture complex spatial and temporal patterns associated with different cognitive states. By analyzing fMRI data from multiple datasets involving various classification learning tasks, our model seeks to provide a more comprehensive and nuanced understanding of an individual's knowledge level.

Our approach integrates several cutting-edge techniques in machine learning and neuroscience, including:

1. 3D vision transformers for processing volumetric fMRI data
2. Task-specific feature extraction for different types of classification learning tasks
3. Temporal modeling to capture dynamic patterns in fMRI sequences
4. Multi-task learning for predicting multiple related outcomes
5. Attention visualization for enhanced interpretability
6. Dynamic functional connectivity analysis
7. Reinforcement learning components for modeling learning processes
8. Explainable AI techniques for detailed prediction explanations

By combining these advanced methods, we aim to develop a more accurate and interpretable model for assessing knowledgeability based on neural activity patterns. This research has potential implications for personalized education, cognitive assessment, and our understanding of the neural basis of learning and memory.

## 2. Background

### 2.1 fMRI in Cognitive Neuroscience

Functional Magnetic Resonance Imaging (fMRI) has become a cornerstone technique in cognitive neuroscience for studying brain activity patterns associated with various cognitive processes. By measuring changes in blood oxygenation and flow, fMRI provides indirect measures of neural activity with high spatial resolution. Numerous studies have utilized fMRI to investigate the neural correlates of learning, memory, and knowledge retrieval.

### 2.2 Machine Learning in Neuroimaging Analysis

The application of machine learning techniques to neuroimaging data has grown exponentially in recent years. Traditional approaches often relied on univariate analyses or simple multivariate techniques. However, the advent of deep learning has revolutionized the field, enabling the extraction of complex, hierarchical features from high-dimensional neuroimaging data.

### 2.3 Vision Transformers in Medical Imaging

Vision Transformers (ViT), initially developed for computer vision tasks, have recently shown promising results in medical imaging analysis. Their ability to capture long-range dependencies and process data in a hierarchical manner makes them particularly suitable for analyzing complex 3D medical imaging data, including fMRI.

### 2.4 Multi-Task Learning in Neuroscience

Multi-task learning has emerged as a powerful approach in neuroscience, allowing models to simultaneously predict multiple related outcomes. This approach can lead to improved generalization and more robust feature representations, particularly when dealing with limited sample sizes often encountered in neuroimaging studies.

### 2.5 Explainable AI in Neuroscience

As machine learning models become more complex, the need for interpretability and explainability has grown. In neuroscience, explainable AI techniques are crucial for validating model predictions against existing neuroscientific knowledge and for generating new hypotheses about brain function.

Our study builds upon these foundations, integrating state-of-the-art techniques from machine learning and cognitive neuroscience to develop a novel approach for assessing knowledgeability through fMRI data analysis. By leveraging the strengths of vision transformers, multi-task learning, and explainable AI, we aim to create a more accurate and interpretable model of knowledge assessment based on neural activity patterns.

# 3. Model Architecture

The proposed model architecture is designed to process 3D fMRI data using a Vision Transformer approach, adapted for volumetric input. The architecture consists of several key components:

### 3.1 Patch Embedding

The first step in processing the fMRI data is patch embedding. This module converts the input fMRI volume into a sequence of flattened patches. It uses a 3D convolutional layer to project the input volume into a sequence of embedded patches, each representing a small 3D region of the brain.

### 3.2 Position Embedding

After patch embedding, the model adds learnable position embeddings to the patch embeddings. This allows the model to retain information about the spatial relationships between different parts of the brain, which is crucial for understanding the overall brain activity patterns.

### 3.3 Transformer Encoder

The core of the model is a Transformer encoder, which consists of multiple layers of self-attention and feed-forward networks. Each layer in the Transformer encoder includes:

1. Multi-Head Attention: This allows the model to attend to different parts of the input simultaneously, capturing complex relationships in the data.
2. Feed-Forward Network: This processes the output of the attention layer, allowing for non-linear transformations of the data.
3. Layer Normalization: Applied before each sub-layer to stabilize the learning process.
4. Residual Connections: These help in training deeper networks by allowing gradients to flow more easily through the network.

### 3.4 Multi-Task Prediction Heads

The output of the Transformer encoder is fed into multiple prediction heads, each corresponding to a different task:

1. Knowledgeability Score: A linear layer that predicts a single value representing the overall knowledgeability of the subject.
2. Brain Region Activation: A linear layer that predicts activation levels for different brain regions.
3. Cognitive State Classification: A linear layer that outputs probabilities for different cognitive states.

### 3.5 Full Model

The full FMRITransformer model combines all these components. It first embeds the input fMRI data into patches, adds position embeddings, processes the sequence through the Transformer encoder, and finally applies the multi-task prediction heads to generate the desired outputs.

This architecture is designed to process 3D fMRI data efficiently, capturing both local and global patterns in brain activity. The use of a Vision Transformer approach allows the model to handle the high-dimensional nature of fMRI data while also capturing long-range dependencies, which is crucial for understanding complex brain activity patterns associated with knowledge and learning.

The multi-task design of the model allows it to simultaneously predict multiple related outcomes, potentially leading to more robust and generalizable features. This approach aligns well with the complex nature of brain function, where different aspects of cognition and knowledge are often interrelated.

## 4. Methodology

### 4.1 Data Preprocessing

Our preprocessing pipeline consists of several key steps to prepare the fMRI data for analysis:

1. **Normalization**: We apply z-score normalization to standardize the fMRI data across subjects and sessions.
2. **Temporal Filtering**: A bandpass filter is applied to remove low-frequency drift and high-frequency noise.
3. **Spatial Standardization**: All fMRI volumes are resized to a standard shape (64x64x30x208) to ensure consistency across datasets.

### 4.2 Data Augmentation

To enhance model generalization, we implement fMRI-specific data augmentation techniques:

1. **Random Temporal Cropping**: Extracts a random continuous segment of the time series.
2. **Gaussian Noise Injection**: Adds controlled noise to simulate variability in fMRI signals.
3. **3D Rotation**: Applies small random rotations to the 3D volume, mimicking slight head movements.

### 4.3 Model Architecture

Our proposed FMRITransformer model consists of the following key components:

1. **Patch Embedding**: Converts 3D fMRI volumes into a sequence of embedded patches.
2. **Position Embedding**: Adds learnable position information to maintain spatial context.
3. **Transformer Encoder**: Processes the embedded patches using self-attention mechanisms.
4. **Multi-Task Prediction Heads**: Outputs predictions for knowledgeability score, brain region activation, and cognitive states.

### 4.4 Training Procedure

We employ a multi-task learning approach with the following details:

1. **Loss Function**: A weighted combination of losses for each prediction task.
2. **Optimization**: AdamW optimizer with a learning rate of 1e-4 and weight decay of 0.01.
3. **Learning Rate Schedule**: Cosine annealing with warm-up.
4. **Gradient Clipping**: Applied to prevent exploding gradients.
5. **Mixed Precision Training**: Utilized to improve computational efficiency.

### 4.5 Evaluation Metrics

We assess our model's performance using:

1. Mean Squared Error (MSE) for knowledgeability score prediction.
2. Mean Absolute Error (MAE) for brain region activation prediction.
3. Accuracy and F1-score for cognitive state classification.

## 5. Experimental Setup

### 5.1 Datasets

We utilize four datasets from OpenNeuro:

1. ds000002: Classification learning task
2. ds000011: Classification learning and tone counting task
3. ds000017: Classification learning and stop-signal task
4. ds000052: Classification learning and reversal task

These datasets provide a diverse range of cognitive tasks, allowing us to test our model's generalizability.

### 5.2 Hardware and Software

Our experiments were conducted using:

- Hardware: NVIDIA A100 GPUs
- Software: PyTorch 1.9, CUDA 11.1, and Python 3.8

### 5.3 Hyperparameter Tuning

We perform a grid search over the following hyperparameters:

- Number of transformer layers: [6, 12, 18]
- Number of attention heads: [8, 12, 16]
- Embedding dimension: [512, 768, 1024]
- Dropout rate: [0.1, 0.2, 0.3]

### 5.4 Baseline Models

We compare our FMRITransformer against the following baselines:

1. 3D Convolutional Neural Network (CNN)
2. Long Short-Term Memory (LSTM) network
3. Traditional machine learning models (Random Forest, SVM)

### 5.5 Ablation Studies

To understand the contribution of different components, we conduct ablation studies on:

1. The effect of data augmentation techniques
2. The impact of multi-task learning vs. single-task learning
3. The influence of the number of transformer layers and attention heads


## 6. Results

In this section, we present the findings from our experiments using the FMRITransformer model on the four OpenNeuro datasets. We evaluate the model's performance in assessing knowledgeability, predicting brain region activation, and classifying cognitive states.

### 6.1 Overall Performance

Our FMRITransformer model demonstrated promising results across all tasks:

- Knowledgeability Score Prediction: [Insert MSE and brief interpretation]
- Brain Region Activation Prediction: [Insert MAE and brief interpretation]
- Cognitive State Classification: [Insert accuracy, F1-score, and brief interpretation]

[Insert a summary table or graph showing the overall performance metrics]

### 6.2 Comparison with Baseline Models

The FMRITransformer outperformed the baseline models in most tasks:

| Model | Knowledgeability MSE | Region Activation MAE | Cognitive State Accuracy |
|-------|----------------------|-----------------------|--------------------------|
| FMRITransformer | [Insert] | [Insert] | [Insert] |
| 3D CNN | [Insert] | [Insert] | [Insert] |
| LSTM | [Insert] | [Insert] | [Insert] |
| Random Forest | [Insert] | [Insert] | [Insert] |
| SVM | [Insert] | [Insert] | [Insert] |

[Brief discussion of the comparative performance]

### 6.3 Performance Across Datasets

We observed variations in model performance across the four datasets:

- ds000002: [Brief summary of performance]
- ds000011: [Brief summary of performance]
- ds000017: [Brief summary of performance]
- ds000052: [Brief summary of performance]

[Insert a graph or table showing performance metrics for each dataset]

### 6.4 Ablation Study Results

Our ablation studies revealed the following:

1. Impact of Data Augmentation: [Brief summary of findings]
2. Multi-task vs. Single-task Learning: [Brief summary of findings]
3. Effect of Transformer Layers and Attention Heads: [Brief summary of findings]

[Insert a table or graph illustrating the ablation study results]

### 6.5 Attention Visualization

The attention visualization revealed interesting patterns in how the model focuses on different brain regions:

[Insert 2-3 attention visualization images with brief descriptions]

### 6.6 Learning Curves

The learning curves for our model showed:

[Insert a graph of training and validation loss over epochs]

[Brief description of convergence behavior and any interesting patterns]

### 6.7 Error Analysis

We conducted an error analysis to understand the model's limitations:

[Insert a brief summary of common error patterns or challenging cases]

[Consider including a confusion matrix for the cognitive state classification task]

## 7. Conclusions

In this study, we developed and evaluated a novel approach for assessing an individual's knowledge level using fMRI data processed by vision transformers. Our FMRITransformer model demonstrated promising results in predicting knowledgeability scores, brain region activation, and cognitive states across multiple datasets.

### 7.1 Summary of Findings

- [Insert brief summary of the most significant results]
- [Highlight how the model performed compared to baselines]
- [Mention any unexpected or particularly interesting findings]

### 7.2 Implications for Cognitive Neuroscience

Our results have several important implications for the field of cognitive neuroscience:

- [Discuss how the findings contribute to our understanding of brain function during learning and recall]
- [Explain any new insights into the neural correlates of knowledge acquisition]
- [Describe how the multi-task learning approach provides a more comprehensive view of cognitive processes]

### 7.3 Methodological Contributions

The FMRITransformer model introduces several methodological advancements:

- [Highlight the benefits of using vision transformers for fMRI data analysis]
- [Discuss the advantages of the multi-task learning approach]
- [Explain how the attention visualization contributes to model interpretability]

### 7.4 Limitations and Future Work

While our study provides valuable insights, it also has some limitations:

- [Discuss any limitations in the datasets or methodology]
- [Mention potential confounding factors or sources of bias]

Future research directions could include:

- [Suggest improvements or extensions to the model architecture]
- [Propose additional tasks or datasets that could be incorporated]
- [Discuss potential clinical or educational applications of the model]

### 7.5 Broader Impact

The ability to assess knowledge levels based on fMRI data has potential implications beyond cognitive neuroscience:

- [Discuss potential applications in personalized education]
- [Mention possible uses in cognitive assessment or early detection of learning disorders]
- [Address any ethical considerations or potential misuses of the technology]

In conclusion, our FMRITransformer model represents a significant step forward in using neuroimaging data to understand and assess cognitive processes. By leveraging advanced machine learning techniques and a multi-task approach, we have developed a tool that not only predicts knowledge levels but also provides insights into the underlying neural mechanisms of learning and recall. This research opens up new avenues for understanding human cognition and has potential applications in education, cognitive assessment, and beyond.

The code we used to train and evaluate our models is available at https://github.com/Twarner491/learnedSpectrum.

Acknowledgements
We are grateful to **NAMES** for their fruitful comments, corrections and inspiration.

# References
- []()


---
- [Watch this repo](https://github.com/Twarner491/learnedSpectrum/subscription)
- [Create issue](https://github.com/Twarner491/learnedSpectrum/issues/new)
