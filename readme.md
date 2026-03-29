<div align="center">

# SSVEP Brain-Computer Interface Classification

**IEEE 2025 Hackathon | Team G18**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MNE](https://img.shields.io/badge/MNE-EEG_Processing-5C9DC0?style=for-the-badge)](https://mne.tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Classifying visual attention through brain signals using deep learning**

[Overview](#overview) | [Features](#features) | [Models](#models) | [Installation](#installation) | [Usage](#usage) | [Results](#results) | [Contributing](#contributing) | [Team](#team)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Running the Notebook](#running-the-notebook)
  - [Using Pre-generated Data](#using-pre-generated-data)
- [Project Architecture](#project-architecture)
  - [Repository Structure](#repository-structure)
  - [Data Format](#data-format)
  - [Pipeline Overview](#pipeline-overview)
- [Models](#models)
  - [TinyEEGNet](#tinyeegnet)
  - [Time-Frequency Two-Branch CNN](#time-frequency-two-branch-cnn)
  - [SVM Baseline](#svm-baseline)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Team](#team)
- [References](#references)
- [License](#license)

---

## Overview

This project implements a **Brain-Computer Interface (BCI)** system that classifies **Steady-State Visual Evoked Potentials (SSVEPs)** from EEG signals. When a user focuses on a flickering visual stimulus, their brain produces electrical responses at the same frequency. Our system detects and classifies these responses to determine which stimulus the user is attending to.

### What is SSVEP?

SSVEP is a neural response elicited when a person focuses on a visual stimulus flickering at a constant frequency. The brain's visual cortex generates electrical activity at the same frequency as the stimulus, which can be detected through EEG electrodes. This makes SSVEP an excellent paradigm for BCI applications because:

- **High signal-to-noise ratio** compared to other BCI paradigms
- **Minimal user training required**
- **Fast communication rates** possible
- **Robust across different users**

### Applications

| Domain | Use Case |
|--------|----------|
| **Assistive Technology** | Communication devices for paralyzed patients |
| **Neuroprosthetics** | Control of robotic limbs and wheelchairs |
| **Gaming** | Hands-free game control |
| **Smart Home** | Brain-controlled device operation |
| **Research** | Cognitive neuroscience studies |

---

## Features

| Category | Description |
|----------|-------------|
| **Signal Processing** | Band-pass filtering (1-40 Hz), notch filtering (50 Hz), epoching |
| **Data Augmentation** | Sliding window segmentation with configurable overlap |
| **Deep Learning** | Custom CNN architectures optimized for EEG classification |
| **Visualization** | Comprehensive plotting tools for signal analysis |
| **Reproducibility** | Pre-generated datasets and documented preprocessing |

### Key Capabilities

- **Multi-frequency classification**: Distinguishes between 4 SSVEP frequencies (9, 10, 12, 15 Hz)
- **Real-time ready**: Sliding window approach enables responsive classification
- **Flexible window sizes**: Supports 0.2s to 2.0s analysis windows
- **Cross-subject analysis**: Data from multiple subjects included

---

## How It Works

```
Raw EEG Signal → Preprocessing → Epoching → Sliding Windows → Neural Network → Classification
     |               |              |              |                |              |
  .mat files    Band-pass +     Extract       Augment data      TinyEEGNet    Predict which
  from device   Notch filter    stimulus      with overlap      or Two-Branch  frequency user
                                periods                         CNN            is focusing on
```

### Signal Flow

1. **Data Acquisition**: EEG recorded while subject views flickering LEDs at 9, 10, 12, and 15 Hz
2. **Preprocessing**: Remove noise with band-pass (1-40 Hz) and notch (50 Hz) filters
3. **Epoching**: Extract time segments corresponding to each stimulus presentation
4. **Segmentation**: Create overlapping windows for data augmentation
5. **Classification**: Neural network predicts which frequency the user attended to

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/anaya33/ssvep-bci-classification.git
cd ssvep-bci-classification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `scipy` | Scientific computing and signal processing |
| `matplotlib` | Visualization |
| `scikit-learn` | Machine learning utilities and SVM baseline |
| `mne` | EEG/MEG data processing |
| `torch` | Deep learning framework |
| `ipykernel` | Jupyter notebook support |

---

## Usage

### Quick Start

1. Open the main notebook:
   ```bash
   jupyter notebook final_code/SSVEP_BCI_Classification_G18.ipynb
   ```

2. Select a Python kernel with dependencies installed

3. Run cells sequentially from top to bottom

### Running the Notebook

The notebook is organized into clearly labeled sections:

| Section | Description |
|---------|-------------|
| **Data Loading** | Load raw .mat files and create MNE objects |
| **Preprocessing** | Apply filters and extract epochs |
| **Sliding Windows** | Generate augmented training data |
| **Model Training** | Train TinyEEGNet or Two-Branch CNN |
| **Evaluation** | Assess model performance |
| **Visualization** | Plot signals and results |

### Using Pre-generated Data

Skip preprocessing by using pre-computed datasets:

```python
import numpy as np

# Load pre-generated sliding window data
data = np.load('final_code/training_data/epochs2_sliding_window_subject_1_1.npz')
X = data['X']  # Shape: (n_windows, n_channels, window_samples)
y = data['y']  # Shape: (n_windows,)
```

Available window sizes: `0.2s`, `0.5s`, `1.0s`, `1.5s`, `2.0s`

---

## Project Architecture

### Repository Structure

```
ieee2025hackathon_g18team/
|
|-- final_code/
|   |-- SSVEP_BCI_Classification_G18.ipynb    # Main notebook (use this)
|   |-- static/                                # Raw EEG recordings
|   |   |-- subject_1_fvep_led_training_1.mat
|   |   |-- subject_1_fvep_led_training_2.mat
|   |   |-- subject_2_fvep_led_training_1.mat
|   |   |-- subject_2_fvep_led_training_2.mat
|   |
|   |-- training_data/                         # Pre-generated datasets
|       |-- epochs{window_size}_sliding_window_subject_{id}_{session}.npz
|
|-- first_tests/                               # Early prototypes (reference only)
|
|-- presentation/
|   |-- SSVEP-18.mp4                          # Project demo video
|
|-- requirements.txt                           # Python dependencies
|-- readme.md                                  # This file
```

### Data Format

#### Raw Data (.mat files)

| Channel | Description |
|---------|-------------|
| 0 | Time stamps |
| 1-8 | EEG channels (occipital region) |
| 9 | Trigger signal |
| 10 | LDA channel |

#### Processed Data (.npz files)

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(n_windows, 8, samples)` | EEG data (8 channels) |
| `y` | `(n_windows,)` | Labels (0-3 for 4 frequencies) |

### Pipeline Overview

#### 1. Data Loading and Preprocessing

```python
# Create MNE Raw object with 11 channels
raw = mne.io.RawArray(data, info)

# Apply filters
raw.filter(l_freq=1.0, h_freq=40.0)  # Band-pass
raw.notch_filter(freqs=50.0)          # Remove power line noise
```

#### 2. Epoching

```python
# Extract epochs around trigger events
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=2.0)
```

#### 3. Sliding Window Segmentation

```python
# Generate overlapping windows for data augmentation
window_size = 2.0  # seconds
step_size = 0.2    # seconds (80% overlap)
```

#### 4. Model Training

```python
# Train neural network
model = TinyEEGNet(n_channels=8, n_classes=4)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
```

---

## Models

### TinyEEGNet

A lightweight 1D CNN designed for efficient EEG classification.

**Architecture:**

```
Input (8 channels x samples)
    |
Conv1D (8 -> 16 filters, kernel=3)
    |
BatchNorm1D
    |
ReLU
    |
GlobalAveragePooling
    |
Linear (16 -> 4 classes)
    |
Output (4 class probabilities)
```

**Characteristics:**
- Minimal parameters for fast inference
- Suitable for real-time applications
- Works best with longer windows (>1.5s)

### Time-Frequency Two-Branch CNN

A novel architecture combining temporal and spectral features.

**Architecture:**

```
Input (8 channels x samples)
    |
    +------------------+
    |                  |
Time Branch        Frequency Branch
(1D Conv)          (STFT -> 2D Conv)
    |                  |
    +------------------+
            |
    Feature Fusion
            |
    Classification Head
            |
    Output (4 classes)
```

**Characteristics:**
- Captures both time-domain and frequency-domain patterns
- Higher accuracy than TinyEEGNet
- Requires more computational resources
- Best performance with 1.5-2.0s windows

### SVM Baseline

A traditional machine learning approach for comparison.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Flatten and standardize
X_flat = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# Train SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
```

---

## Results

### Model Comparison

| Model | Window Size | Accuracy | Notes |
|-------|-------------|----------|-------|
| TinyEEGNet | 1.0s | ~25% | Near chance (4 classes) |
| TinyEEGNet | 2.0s | ~40% | Improved with longer windows |
| Two-Branch CNN | 1.5s | ~65% | Significant improvement |
| Two-Branch CNN | 2.0s | ~75% | Best performance |
| SVM Baseline | 2.0s | ~45% | Traditional ML comparison |

### Key Findings

- **Window length matters**: Longer windows (1.5-2.0s) significantly improve accuracy
- **Frequency features help**: The two-branch architecture outperforms time-only models
- **Data augmentation is crucial**: Sliding windows with overlap improve generalization

---

## Configuration

### Adjustable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t_epoch` | 2.0s | Epoch duration |
| `window_size` | 2.0s | Sliding window length |
| `step_size` | 0.2s | Window step (overlap = 1 - step/window) |
| `l_freq` | 1.0 Hz | High-pass filter cutoff |
| `h_freq` | 40.0 Hz | Low-pass filter cutoff |
| `notch_freq` | 50.0 Hz | Notch filter frequency |

### Adapting to Your Data

1. **Add new recordings**: Place `.mat` files in `final_code/static/`

2. **Update file paths**: Modify glob patterns in data loading cells
   ```python
   mat_files = glob.glob('final_code/static/your_pattern_*.mat')
   ```

3. **Adjust trigger mapping**: If your protocol differs, update event mapping
   ```python
   # Default: [15, 12, 10, 9] Hz in order of appearance
   freq_order = [15, 12, 10, 9]
   ```

4. **Tune window parameters**: Trade off latency vs accuracy
   ```python
   window_size = 1.5  # Shorter = faster, longer = more accurate
   step_size = 0.1    # Smaller = more data, larger = less overlap
   ```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Import errors** | Ensure all dependencies installed: `pip install -r requirements.txt` |
| **Path not found** | Check working directory; paths are relative to repo root |
| **CUDA out of memory** | Reduce batch size or use CPU: `device = 'cpu'` |
| **Low accuracy** | Try longer window sizes (1.5-2.0s) |
| **Trigger detection fails** | Verify trigger channel index and threshold |

### Path Issues

Some early cells may reference Colab-style paths (`/content/`). For local execution, ensure paths point to:
- Raw data: `final_code/static/`
- Processed data: `final_code/training_data/`

### GPU Support

PyTorch automatically uses CUDA if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## Contributing

Contributions are welcome! Here's how to help:

### Ways to Contribute

- **Bug fixes**: Found an issue? Submit a pull request
- **New models**: Implement additional architectures (EEGNet, Transformer, etc.)
- **Documentation**: Improve explanations or add tutorials
- **Visualization**: Create better plots or interactive dashboards
- **Optimization**: Improve training speed or model efficiency

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Test thoroughly
5. Submit a pull request with a clear description

### Ideas for Contributors

| Difficulty | Task |
|------------|------|
| Easy | Add more visualization functions |
| Easy | Improve code documentation |
| Medium | Implement EEGNet architecture |
| Medium | Add cross-validation support |
| Medium | Create a training script (separate from notebook) |
| Advanced | Implement online/streaming classification |
| Advanced | Add transfer learning between subjects |
| Advanced | Build a real-time demo application |

---

## Team

### Team G18 Members

| Name | GitHub |
|------|--------|
| **Haocheng Wu** | [@TedHaochengWu](https://github.com/TedHaochengWu) |
| **Mohammadreza Behbood** | [@mudcontract](https://github.com/mudcontract) |
| **Soukaina Hamou** | [@SoukainaHAMOU](https://github.com/SoukainaHAMOU) |
| **Nathan Yu** | [@Littnatenate](https://github.com/Littnatenate) |
| **Jeronimo Sanchez Santamaria** | [@JeronimoSantamaria](https://github.com/JeronimoSantamaria) |
| **Flora Santos** | - |
| **Anaya Yorke** | [@anaya33](https://github.com/anaya33) |

### Project Demo

Watch our complete project walkthrough: `presentation/SSVEP-18.mp4`

---

## References

### Documentation

- [MNE-Python](https://mne.tools/stable/) - EEG/MEG analysis toolkit
- [PyTorch](https://pytorch.org/docs/) - Deep learning framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library

### Papers

- Vialatte, F. B., et al. "Steady-state visually evoked potentials: focus on essential paradigms and future perspectives." Progress in neurobiology (2010)
- Lawhern, V. J., et al. "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." Journal of neural engineering (2018)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**IEEE 2025 Hackathon | Team G18**

[Back to Top](#ssvep-brain-computer-interface-classification)

</div>

