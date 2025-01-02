# Arrhythmia Detection Project

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