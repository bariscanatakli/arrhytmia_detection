# PTB-XL Run Summary (Octave)

Run command:
```bash
octave --persist --eval "addpath('octave'); pkg load io; ptbxl_data_analysis"
```

Dataset location: `dataset/` (flat layout detected automatically)

## Key metrics (age <= 89 filter)
- Records: 21,506; unique patients: 18,612
- Age range: 2–89; mean/median age: 59.5 / 61.0
- Sex split: Male 10,220; Female 11,286
- Recording: 10s, 12 leads, 500 Hz + 100 Hz versions
- CSV size: 6.29 MB; missing cells (all columns): 207,475

## Label snapshots
- Approx normal vs abnormal (presence of `NORM`): Normal 9,514; Abnormal 12,285
- Most frequent SCP codes (top 10 by records):
  - SR 16,748 (76.83%)
  - NORM 9,514 (43.64%)
  - ABQRS 3,327 (15.26%)
  - IMI 2,676 (12.28%)
  - ASMI 2,357 (10.81%)
  - LVH 2,132 (9.78%)
  - NDT 1,825 (8.37%)
  - LAFB 1,623 (7.45%)
  - AFIB 1,514 (6.95%)
  - ISC_ 1,272 (5.84%)

Arrhythmia-focused classes (10-class set, counts include multi-label overlap):
- SR 16,748 (76.83%), AFIB 1,514 (6.95%), STACH 826 (3.79%), SARRH 772 (3.54%)
- PVC 1,143 (5.24%), PAC 398 (1.83%), AFLT 73 (0.33%), SBRAD 637 (2.92%), SVTAC 27 (0.12%), NORM 9,514 (43.64%)

## Patient-level + missingness highlights
- Patients with AFIB: 1,245 (6.60%); with PVC: 1,057 (5.60%); with NORM: 8,896 (47.15%)
- Weight present in 43.22% of records (mean age 60.68); missing in 56.78% (mean age 64.36)

## Figures
Generated PNGs saved to `analysis_results/` (repo root):
- `octave_age_distribution.png`, `octave_sex_distribution.png`, `octave_normal_vs_abnormal.png`, `octave_arrhythmia_class_counts.png`
- `octave_arrhythmia_age_stats.png`, `octave_arrhythmia_sex_distribution.png`, `octave_age_group_sex_distribution.png`, `octave_scp_cooccurrence_heatmap.png`, `octave_diagnostic_class_distribution.png`

If you ran Octave from a different working directory and don’t see the images, ensure the output path points to the repo’s `analysis_results/`.
