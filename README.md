# MFPIT — Multi-Factor Physics-Informed Transformer
## Geospatial Intelligence-Driven Hydrological Modelling, Indus River Basin (Sindh)

### Execution Order
```
python MFPIT_step1_create_dirs.py
python gee/gee_auth_check.py
python gee/gee_data_extractor.py --dataset static
python gee/gee_data_extractor.py --dataset all --years 2001 2023
python models/train.py --model all
python models/verify.py
```

### Study Area
Bounding box: 67.0–69.5°E, 23.5–28.0°N (Sukkur → Arabian Sea Delta, Sindh, Pakistan)

### Requirements
See requirements.txt — Python 3.9+, CUDA optional but recommended for training.

### Key Outputs
- `data/DOWNLOAD_MANIFEST.csv` — per-file data quality summary
- `models/checkpoints/mfpit_full_best.pth` — best MFPIT checkpoint
- `MFPIT_benchmark_table.csv` — all model metrics
- `figures/` — 16 publication-quality figures (300 DPI)
- `paper/main.pdf` — compiled Q1 journal paper
