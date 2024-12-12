# OpenFMRI Classification Learning Datasets

This README provides a comprehensive overview of four classification learning datasets from the OpenFMRI project. These datasets serve as crucial resources for researchers investigating cognitive processes underlying learning, decision-making, and task-switching paradigms.

## Datasets

### 1. [Classification Learning (ds000002)](https://openfmri.org/dataset/ds000002/)

#### Participants
17 right-handed healthy English-speaking subjects participated in feedback-driven classification learning tasks.

#### Task Design
Subjects completed two distinct paradigms:
1. Probabilistic Classification (PROB)
2. Deterministic Classification (DET)

The experiment concluded with mixed blocks implementing an event-related design.

#### Procedure
Pure blocks consisted of 10 cycles, each containing 5 classification trials followed by 3 baseline trials. Mixed blocks presented 100 stimuli (50 PROB, 50 DET). Weather prediction via card combinations served as the primary task. Stimulus presentation lasted 4s in pure blocks, 2.5s in mixed blocks. Feedback appeared exclusively in pure blocks.

#### MRI Acquisition
Data collection utilized a 3T Siemens Allegra scanner:
- 180 functional T2*-weighted EPI
- TR=2s, TE=30ms, flip angle=90°, FOV=200mm
- 33 slices, 4mm thickness

### 2. [Classification Learning and Tone-Counting (ds000011)](https://openfmri.org/dataset/ds000011/)

#### Participants
14 subjects underwent fMRI scanning during classification tasks.

#### Task Design
The experiment implemented dual classification problems:
1. Single-Task (ST) condition
2. Dual-Task (DT) condition w/ concurrent tone-counting

#### Procedure
Training phase employed trial-error feedback mechanisms. Probe phase utilized mixed event-related fMRI without feedback. ST conditions dominated probe phase, supplemented by tone-counting localizer scans.

### 3. [Classification Learning and Stop-Signal (ds000017)](https://openfmri.org/dataset/ds000017/)

#### Participants
N=8 healthy subjects (mean age 38.9±10.1 years; 1 female)

#### Task Design
Parallel experimental paradigms:
1. Probabilistic Classification Learning (PCL)
2. Cued Response-Inhibition (stop-signal)

#### Procedure
PCL implemented 50 trials + 30 baseline trials (10 cycles [5 PCL + 3 baseline]). Stop-signal task comprised 2-3 runs of 256 trials each. PCL protocol matched ds000002 specifications. Stop-signal required arrow direction responses with inhibition cues.

### 4. [Classification Learning and Reversal (ds000052)](https://openfmri.org/dataset/ds000052/)

#### Task Design
Four blocks of event-related PCL:
- Blocks 1-2: Original reward contingencies
- Blocks 3-4: Reversed reward contingencies

## Data Processing Pipeline

### Preprocessing

1. `preprocess_volume`:
   - Dimension validation
   - Spatial resizing via zoom
   - Intensity normalization
   - Motion correction
   - Slice timing correction

2. `normalize_temporal_resolution`:
   Ensures temporal consistency through TR resampling

3. `FMRIAugmentor`:
   - Temporal masking
   - Spatial masking
   - Noise injection
   - Signal augmentation

### Data Loading Architecture

1. `create_dataloaders`: Implements train/validation splitting with appropriate collation functions

2. `collate_fn`/`collate_variable_length`: Manages batch creation for variable-length sequences

3. `FMRIDataset`: 
   - BIDS format management
   - NIFTI validation/loading
   - AAL region extraction
   - Wavelet decomposition for temporal patterns
   - Memory-efficient caching system

4. Feature Extraction Protocols:
   - `extract_aal_regions`: Regional activation analysis
   - `extract_temporal_patterns`: Dynamic pattern recognition

5. `BIDSManager`: Dataset organization adhering to BIDS specification

## Methodological Considerations

- Consistent hardware (3T Siemens Allegra) facilitates cross-study comparison
- Researchers must account for acquisition parameter variations
- Pipeline handles spatial/temporal normalization, feature extraction, efficient loading
- Data quality metrics implemented throughout processing stream

## References

1. Aron, A. R., Gluck, M. A., & Poldrack, R. A. (2006). Long-term test-retest reliability of functional MRI in classification learning. NeuroImage, 29(3), 1000-1006.

2. Knowlton, B. J., Mangels, J. A., & Squire, L. R. (1996). A neostriatal habit learning system in humans. Science, 273(5280), 1399-1402.

3. Poldrack, R. A., et al. (2001). Interactive memory systems in the human brain. Nature, 414(6863), 546-550.