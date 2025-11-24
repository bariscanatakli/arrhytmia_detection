# Arrhythmia Detection Project

For a quick start guide, see `docs/getting_started.md`.

## Overview

This project aims to detect cardiac arrhythmias using deep learning techniques applied to ECG signals. The model is trained on the PTB-XL ECG database, which contains 21,837 clinical 12-lead ECG records from 18,885 patients. The goal is to provide a reliable tool for early detection of arrhythmias, which can be critical for patient care.

## Features

- **Deep Learning Model**: Utilizes state-of-the-art neural network architectures for accurate arrhythmia detection.
- **Comprehensive Dataset**: Trained on the PTB-XL database, ensuring a wide variety of ECG patterns.
- **Model Evaluation**: Includes scripts for thorough evaluation of model performance.
- **Data Analysis**: Provides notebooks for exploratory data analysis and visualization.

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Dataset Installation

1. Download the PTB-XL database from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2. Create a folder named `dataset` in the `arrhytmia_detection` directory.
3. Extract the downloaded database files into the `dataset` folder.
4. Ensure your folder structure matches the following:

```plaintext
.
├── arrhytmia_detection/          # Main project code
│   ├── dataset/                  # PTB-XL ECG database
│   │   ├── records100/           # 100Hz ECG recordings
│   │   └── records500/           # 500Hz ECG recordings
│   ├── model_test/               # Model evaluation scripts
│   ├── best_model.keras          # Trained model weights
│   └── data_analysis.ipynb       # Data exploration notebook
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bariscanatakli/arrhytmia_detection.git
   cd arrhytmia_detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Training the Model

To train the model, run the following command:
```bash
python train_model.py
```

### MATLAB (Alternative Implementation)

This repository also includes a MATLAB reimplementation under `matlab/` for users who prefer MATLAB.

Prerequisites:
- MATLAB R2021b or newer recommended
- Deep Learning Toolbox
- WFDB Toolbox (for `rdsamp`) or equivalent access to PTB‑XL WFDB files

Steps:
1. Ensure the PTB‑XL dataset is downloaded and placed under `dataset/` as described above.
2. In MATLAB, add the `matlab/` folder to your path.
3. Run the training script:
   ```matlab
   cd matlab
   train_model
   ```
   - Outputs and checkpoints are written to `matlab_output/`.
   - Configuration (paths, classes, hyperparameters) is in `matlab/config.m`.

Notes:
- The MATLAB pipeline implements a 1D CNN + LSTM with sigmoid outputs and trains with binary cross‑entropy using a custom loop. It is designed for multi‑label classification consistent with the Python version.
- For large‑scale runs, consider adapting `matlab/load_ptbxl.m` to stream data rather than loading all records into memory.

### MATLAB Data Analysis

You can generate dataset analysis figures (class distribution, sample ECG plots, demographics) via:

```matlab
cd matlab
data_analysis
```

Outputs are saved under `analysis_results/` as PNG files.

### GNU Octave (Data Science Pipeline)

If you prefer GNU Octave, there is a lightweight data‑science oriented pipeline under `octave/` which works directly on the PTB‑XL metadata (no deep learning, fast to experiment with).

Prerequisites:
- GNU Octave 6.0 or newer (recommended)
- Octave `io` package (for CSV parsing)

Install the `io` package once inside Octave:
```octave
pkg install -forge io
pkg load io
```

Basic exploration (EDA – metadata only):
```bash
cd octave
octave --persist --eval "ptbxl_basic_eda"
```
- Reads `ptbxl_database.csv` from `dataset/physionet.org/files/ptb-xl/1.0.3`.
- Prints summary statistics (age, sex).
- Plots histograms for age, sex distribution, and an approximate normal/abnormal split based on the `scp_codes` field.

Simple metadata‑only model (logistic regression):
```bash
cd octave
octave --persist --eval "ptbxl_simple_model"
```
- Builds a design matrix from age, sex, height, and weight.
- Creates a binary target: “normal” vs “non‑normal” using the presence of `NORM` in `scp_codes`.
- Trains a logistic regression classifier with gradient descent (implemented in pure Octave).
- Reports test accuracy and a 2×2 confusion matrix (`[TN FP; FN TP]`), plus the learned weights.

Notes:
- This Octave pipeline does **not** touch raw ECG waveforms; it is meant as a quick data science / prototyping environment on top of PTB‑XL metadata.
- For serious arrhythmia work you should combine this with signal‑level features or the existing Python / MATLAB deep learning models.

Octave advanced analysis (richer plots, no model):
```bash
cd octave
octave --persist --eval "ptbxl_data_analysis"
```
- Prints a dataset summary similar to `data_analysis.ipynb` (record/patient counts, age range, gender distribution, missing values, file size).
- Generates basic distribution plots via `ptbxl_basic_eda`.
- Generates advanced plots via `ptbxl_advanced_plots`:
  - Age distribution per arrhythmia class (boxplot).
  - Gender distribution across arrhythmia classes (stacked bar chart).
  - Co-occurrence heatmap for selected SCP codes (e.g. AFIB, PVC, NORM, LVH).

### WSL Setup (Run Windows MATLAB from WSL)

If you are on WSL and `matlab` is not available in the Linux PATH, you can call the Windows MATLAB from WSL using the helper scripts:

1. Make scripts executable:
   ```bash
   chmod +x scripts/run_matlab_*.sh
   ```
2. Optionally set your MATLAB path (if auto-detection fails):
   ```bash
   export MATLAB_EXE="/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe"
   ```
3. Run analysis or training from WSL:
   ```bash
   ./scripts/run_matlab_analysis.sh
   ./scripts/run_matlab_training.sh
   ```

These scripts locate `matlab.exe`, convert the repo path to a Windows path, and run the corresponding MATLAB function via `-batch`.

### Evaluating the Model

To evaluate the model, we have random test script.
```bash
python model_test/random_test.py
```

### Data Analysis

Explore the dataset using the provided Jupyter notebook:
```bash
jupyter notebook data_analysis.ipynb
```

## Contributing

We welcome contributions to improve the project. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The PTB-XL database is provided by PhysioNet.
- Special thanks to all contributors and the open-source community.
