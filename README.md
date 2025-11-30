# Arrhythmia Detection (MATLAB / Octave)

This branch is simplified for MATLAB / GNU Octave users. The focus is on PTB-XL metadata exploration, lightweight classical models, and an interactive dashboard.

Presentations: https://presentations.bariscanatakli.com/ptbxl/ (offline: open `presentation.html` locally; no external assets required)

## Prerequisites
- MATLAB R2021b+ **or** GNU Octave 6+  
- Octave `io` package (for CSV parsing)  
- PTB-XL dataset unpacked under `dataset/physionet.org/files/ptb-xl/1.0.3/` (must include `ptbxl_database.csv` and at least one of `records100/` or `records500/`)  
- Optional: WFDB Toolbox (MATLAB/Octave) for raw signal preview via `rdsamp`

Dataset layout:
```
dataset/
  physionet.org/files/ptb-xl/1.0.3/
    ptbxl_database.csv
    records100/
    records500/
    scp_statements.csv
```
- If your dataset lives elsewhere, set `PTBXL_BASE=/custom/path/to/ptb-xl/1.0.3` before running any Octave scripts.
- All Octave entry points now auto-detect the dataset location with `ptbxl_autodetect_base`:
  - Checks the function argument, then `PTBXL_BASE`, then common relative paths (repo root, `octave/` folder).
  - Errors list exactly what is missing (`ptbxl_database.csv`, `scp_statements.csv`, `records100`/`records500`) to simplify setup.

## Quick start (Octave / MATLAB)
From the repo root:
```octave
addpath('octave'); pkg load io; % in MATLAB omit pkg load
ptbxl_data_analysis        % summary + figures to analysis_results/
% graphics_toolkit qt; ptbxl_dashboard  % interactive dashboard (Octave)
```
- `ptbxl_data_analysis` prints dataset stats (records/patients, age range, sex split, missing values) and saves PNGs for age/sex/arrhythmia distributions and SCP co-occurrence under `analysis_results/`.
- `ptbxl_dashboard` (Octave GUI) offers filters by age/sex/arrhythmia class and exports the same plots; menu items include patient-level summaries and optional WFDB signal preview when `rdsamp` is available.

## Octave install notes (when `octave` is missing)
- Debian/Ubuntu/WSL: `sudo apt-get update && sudo apt-get install -y octave`
- macOS (Homebrew): `brew install octave`
- Windows: prefer MATLAB or install Octave inside WSL using the Ubuntu command above.
- Verify with `octave --version`, then run the commands in the quick start.

## Demo screenshot (add when ready)
Capture the dashboard and drop the image into the repo, then link it here:
```
octave --persist --eval "graphics_toolkit qt; addpath('octave'); pkg load io; ptbxl_dashboard"
```
Take an OS screenshot of the running dashboard and save it as `docs/img/dashboard_demo.png` (create the folder if needed). Add the Markdown below once the file exists:
`![PTB-XL dashboard demo](docs/img/dashboard_demo.png)`

## Lightweight models
- `ptbxl_simple_model`: metadata-only logistic regression (age, sex, height, weight) to separate `NORM` vs non-`NORM`; reports accuracy and a 2×2 confusion matrix.
- `ptbxl_patient_eda`: patient-level counts (records per patient, arrhythmia prevalence) printed to console and optionally plotted.

## Helpful scripts
- `ptbxl_basic_eda`: age/sex distributions, normal vs abnormal split, and top arrhythmia counts.
- `ptbxl_advanced_plots`: arrhythmia age statistics, arrhythmia sex distribution, and SCP co-occurrence heatmap (saved to `analysis_results/`).
- `ptbxl_cooccurrence_matrix`: helper to compute SCP co-occurrence across the selected codes.

## Notes on signals
- For raw ECG previews, install WFDB Toolbox and ensure `WFDBROOT` points to your PTB-XL path. If WFDB is missing, the dashboard attempts a Python `wfdb` fallback when that module is available; otherwise signal preview is skipped.
- All current MATLAB/Octave utilities operate on metadata; model training on signals is intentionally out of scope for this branch.

## Repository layout
- `octave/`: all MATLAB/Octave functions listed above.
- `analysis_results/`: generated PNGs and text summaries.
- Legacy assets kept only for reference: `best_model.keras`, `data_analysis.ipynb`, `model_test/` images. Python training/testing scripts were removed in this MATLAB-focused branch.
