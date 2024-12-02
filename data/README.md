# OpenFMRI Classification Learning Datasets

This README provides a comprehensive overview of four classification learning datasets from the OpenFMRI project, along with details on data loading and preprocessing methods. These datasets offer valuable resources for researchers studying cognitive processes related to learning, decision-making, and task-switching.

## Datasets

### 1. [Classification Learning (ds000002)](https://openfmri.org/dataset/ds000002/)

#### Participants
17 right-handed healthy English-speaking subjects (age range not specified).

#### Task Design
Subjects participated in two types of feedback-driven classification learning tasks:

1. Probabilistic (PROB)
2. Deterministic (DET)

Followed by "mixed blocks" in an event-related design.

#### Procedure
- Pure blocks: 10 cycles of 5 classification trials followed by 3 baseline trials
- Mixed blocks: 100 stimuli (50 PROB, 50 DET)
- Subjects predicted weather (sun or rain) based on card combinations
- Stimulus presentation: 4 seconds (pure blocks), 2.5 seconds (mixed blocks)
- Feedback provided in pure blocks, not in mixed blocks

#### MRI Acquisition
- 3T Siemens Allegra MRI scanner
- 180 functional T2*-weighted echoplanar images (EPI)
- TR = 2s, TE = 30ms, flip angle = 90°, FOV = 200mm, 33 slices, 4mm slice thickness

### 2. [Classification Learning and Tone-Counting (ds000011)](https://openfmri.org/dataset/ds000011/)

#### Participants
14 subjects underwent fMRI scanning while performing classification learning tasks.

#### Task Design
Subjects were trained on two different classification problems:
1. Single-task (ST) condition
2. Dual-task (DT) condition with a concurrent tone-counting task

#### Procedure
- Training phase: Subjects learned categories through trial-by-trial feedback
- Probe phase: Mixed event-related fMRI paradigm without feedback
- All items presented under ST conditions during the probe phase
- Additional tone-counting localizer scan included

#### Tasks
1. Tone counting
2. Single-task weather prediction
3. Dual-task weather prediction
4. Classification probe without feedback

#### MRI Acquisition
- 3T Siemens Allegra head-only MR scanner

### 3. [Classification Learning and Stop-Signal (ds000017)](https://openfmri.org/dataset/ds000017/)

#### Participants
8 healthy subjects (mean age 38.9 ± 10.1 years; 1 female)

#### Task Design
Two distinct tasks:
1. Probabilistic classification learning (PCL)
2. Cued response-inhibition task (stop-signal task)

#### Procedure
- PCL task: 50 PCL trials and 30 baseline trials (10 cycles of 5 PCL + 3 baseline)
- Stop-signal task: 2-3 runs of 256 trials each
- PCL task identical to ds000002
- Stop-signal task: Subjects responded to arrow direction, inhibiting response on specific cues

#### MRI Acquisition
- 3T Siemens Allegra MRI scanner
- 180 functional T2*-weighted echoplanar images (EPI)
- TR = 2s, TE = 30ms, flip angle = 90°, FOV = 200mm, 33 slices, 4mm slice thickness

### 4. [Classification Learning and Reversal (ds000052)](https://openfmri.org/dataset/ds000052/)

#### Task Design
Four blocks of event-related probabilistic classification learning:
1. Two initial blocks with original reward contingencies
2. Two additional blocks with reversed reward contingencies

## Data Loading and Preprocessing

### Preprocessing

1. `preprocess_volume`: Performs robust fMRI preprocessing, including:
   - Dimension validation
   - Spatial resizing using zoom
   - Intensity normalization

2. `normalize_temporal_resolution`: Ensures temporal consistency across datasets by resampling to a target TR (repetition time).

3. `FMRIAugmentor`: Applies fMRI-specific data augmentations:
   - Temporal masking
   - Spatial masking
   - Noise injection

### Data Loading

1. `create_dataloaders`: Creates train and validation data loaders with appropriate splitting and collate functions.

2. `collate_fn` and `collate_variable_length`: Handle batch creation, including proper handling of variable-length sequences and mixed data types.

3. `FMRIDataset`: A custom dataset class that:
   - Manages BIDS-formatted data
   - Validates and loads NIFTI files
   - Extracts AAL (Automated Anatomical Labeling) regions
   - Computes temporal patterns using wavelet decomposition
   - Implements caching for efficient data loading

4. `extract_aal_regions` and `extract_temporal_patterns`: Extract meaningful features from raw fMRI data, including regional activations and temporal dynamics.

5. `BIDSManager`: Handles dataset fetching and organization according to the BIDS (Brain Imaging Data Structure) format.

## Notes

- The consistent use of 3T Siemens Allegra MRI scanners across studies facilitates comparability of results.
- Researchers should be aware of potential differences in acquisition parameters and task designs when comparing across datasets.
- The data loading and preprocessing pipeline is designed to handle the complexities of fMRI data, including spatial and temporal normalization, feature extraction, and efficient data loading.

## References

1. Aron, A. R., Gluck, M. A., and Poldrack, R. A. (2006). Long-term test-retest reliability of functional MRI in a classification learning task. Neuroimage, 29(3):1000–6.

2. Knowlton, B. J., Mangels, J. A., and Squire, L. R. (1996). A neostriatal habit learning system in humans. Science, 273(5280):1399–402.

3. Poldrack, R. A., Clark, J., Paré-Blagoev, E. J., Shohamy, D., Creso Moyano, J., Myers, C., and Gluck, M. A. (2001). Interactive memory systems in the human brain. Nature, 414(6863):546–50.

For more detailed information on each dataset, please refer to the original publications and the OpenFMRI project documentation.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/9534770/23b8b4fb-91ee-4a22-83e9-a630f3fe2d1f/classificationLearningAndStop-Signal.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/9534770/bb6cb383-5edb-4a12-ba4e-cd353a15b921/classificationLearning.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/9534770/d492a1e3-cee2-41c4-b9a9-5000d8ba726c/classificationLearningAndReversal.pdf
