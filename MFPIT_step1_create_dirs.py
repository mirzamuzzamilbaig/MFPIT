"""
MFPIT Pipeline — Step 1: Directory Tree Creation
=================================================
Run this FIRST before any other script.
Creates the full MFPIT_Project/ directory structure inside PhD-Project/.

Usage:
    python MFPIT_step1_create_dirs.py

[CHECKPOINT ✓] will be printed on success.
"""

import os
import json
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG: Set this to your PhD-Project root path
# ──────────────────────────────────────────────
PHD_ROOT = Path(".")          # Change to absolute path if needed
PROJECT_ROOT = PHD_ROOT / "MFPIT_Project"

DIRS = [
    "data/raw/sentinel2",
    "data/raw/chirps",
    "data/raw/terraclimate",
    "data/raw/jrc_water",
    "data/raw/dem",
    "data/raw/modis_et",
    "data/raw/sar",
    "data/raw/jrc_occurrence",
    "data/processed/patches",
    "data/processed/tensors",
    "data/ground_truth",
    "models/checkpoints",
    "gee",
    "figures",
    "paper/sections",
    "logs",
]

def create_project_tree():
    print("=" * 60)
    print("MFPIT Step 1: Creating project directory tree")
    print("=" * 60)

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    for d in DIRS:
        full_path = PROJECT_ROOT / d
        full_path.mkdir(parents=True, exist_ok=True)
        # Placeholder so git tracks the empty folder
        keeper = full_path / ".gitkeep"
        if not keeper.exists():
            keeper.touch()
        print(f"  [OK] {full_path}")

    _init_progress_log()
    _init_readme()

    print()
    print("[CHECKPOINT ✓] Step 1 completed. Project tree created at:")
    print(f"               {PROJECT_ROOT.resolve()}")
    print()

def _init_progress_log():
    log_path = PROJECT_ROOT / "AGENT_PROGRESS_LOG.md"
    if log_path.exists():
        print("  [SKIP] AGENT_PROGRESS_LOG.md already exists.")
        return

    content = f"""# MFPIT Agent Progress Log
Generated: {datetime.now().isoformat()}

---

## Checkpoint History

| Step | Name | Status | Timestamp | Output Files | Key Metrics | Notes |
|------|------|--------|-----------|--------------|-------------|-------|
| 1 | Directory tree creation | ✓ PASS | {datetime.now().strftime('%Y-%m-%d %H:%M')} | MFPIT_Project/ | — | Initialised |

---

## Download Manifest
> Will be populated after Step 6 (verify_downloads)

## Metric Tracker
> Will be populated after Step 13 (verify.py)
"""
    log_path.write_text(content, encoding="utf-8")
    print("  [OK] AGENT_PROGRESS_LOG.md initialised")

def _init_readme():
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        return

    content = """# MFPIT — Multi-Factor Physics-Informed Transformer
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
"""
    readme_path.write_text(content, encoding="utf-8")
    print("  [OK] README.md initialised")

if __name__ == "__main__":
    create_project_tree()
