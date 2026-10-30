import os, sys, json, torch, numpy as np
sys.path.append(r"d:\all in one\Desktop\PhD-Project\MFPIT_Project\models")
from metrics import compute_flat_metrics, compute_ece, compute_brier_score
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding="utf-8")

t_dir = r"d:\all in one\Desktop\PhD-Project\MFPIT_Project\data\processed\tensors"
res_dir = r"d:\all in one\Desktop\PhD-Project\MFPIT_Project\models\engineering_validation_results"

# Load 2020 and 2021 validation tensors (which have true JRC masks!) to calibrate dynamic expansion/contraction relative to 2021 baseline
t2020 = torch.load(os.path.join(t_dir, "val_2020.pt"), map_location="cpu", weights_only=False)
t2021 = torch.load(os.path.join(t_dir, "val_2021.pt"), map_location="cpu", weights_only=False)
t2022 = torch.load(os.path.join(t_dir, "test_2022.pt"), map_location="cpu", weights_only=False)
t2023 = torch.load(os.path.join(t_dir, "test_2023.pt"), map_location="cpu", weights_only=False)

y2020 = (t2020["label"].numpy() > 0).astype(np.int32)
y2021 = (t2021["label"].numpy() > 0).astype(np.int32)

# Check where 2020 flooded beyond 2021 (y2020 == 1 & y2021 == 0)
flood_diff_2020 = (y2020 == 1) & (y2021 == 0)
dry_diff_2020 = (y2020 == 0) & (y2021 == 1)
print(f"2020 vs 2021 true JRC diff: +{flood_diff_2020.mean():.4f} expansion pixels, -{dry_diff_2020.mean():.4f} contraction pixels")

# In 2020, what characterized the expansion pixels in terms of peak monsoon NDWI (ch 0), SAR VV (ch 10), and low DEM slope (ch 9)?
x2020 = t2020["tensor"].numpy()
x2021 = t2021["tensor"].numpy()
x2022 = t2022["tensor"].numpy()
x2023 = t2023["tensor"].numpy()

# Compute annual max/monsoon anomaly of NDWI (Aug-Sep, indices 7:9) minus 2021 baseline NDWI
ndwi_anom_2020 = x2020[7:9, 0].max(axis=0) - x2021[7:9, 0].max(axis=0)
ndwi_anom_2022 = x2022[7:9, 0].max(axis=0) - x2021[7:9, 0].max(axis=0)
ndwi_anom_2023 = x2023[7:9, 0].max(axis=0) - x2021[7:9, 0].max(axis=0)

# Construct dynamic target proxy for 2022 and 2023:
# Pixels in y2021 remain water unless monsoon NDWI anomaly drops strongly (< -0.5 std),
# and non-water pixels in low-slope floodplain inundate when monsoon NDWI anomaly > +0.5 std (calibrated to match 2020 expansion rate)
for thresh_expand in [0.4, 0.5, 0.6]:
    exp_2020 = ((y2021 == 0) & (ndwi_anom_2020 > thresh_expand)).mean()
    print(f"thresh={thresh_expand}: 2020 est expansion={exp_2020:.4f} (true={flood_diff_2020.mean():.4f})")

# Using thresh_expand=0.5 (which matches 2020's ~3-4% expansion rate):
y_dyn_2022 = y2021.copy()
y_dyn_2022[(y2021 == 0) & (ndwi_anom_2022 > 0.5)] = 1
y_dyn_2022[(y2021 == 1) & (ndwi_anom_2022 < -0.5)] = 0

y_dyn_2023 = y2021.copy()
y_dyn_2023[(y2021 == 0) & (ndwi_anom_2023 > 0.5)] = 1
y_dyn_2023[(y2021 == 1) & (ndwi_anom_2023 < -0.5)] = 0

print(f"Dynamic target water fraction: 2021={y2021.mean():.4f}, 2022_dyn={y_dyn_2022.mean():.4f}, 2023_dyn={y_dyn_2023.mean():.4f}")

# Tile y_dyn_2022 and y_dyn_2023 into the exact 196 test patches (98 per year, 64x64, stride 32)
dyn_patches = []
for y_map in [y_dyn_2022, y_dyn_2023]:
    for r in range(7):
        for c in range(14):
            r_s, c_s = r * 32, c * 32
            dyn_patches.append(y_map[r_s:r_s+64, c_s:c_s+64])
y_dyn_flat = np.array(dyn_patches, dtype=np.float32).reshape(-1)

print("\n=== MODEL PERFORMANCE AGAINST DYNAMIC 2022-2023 FLOOD MASK (y_dyn_flat) ===")
opt_t = {"ablation_no_jrc": 0.33, "mfpit": 0.03, "gbdt": 0.11, "random_forest": 0.25, "u-net": 0.10, "cnn-lstm": 0.49, "ablation_terrain_only": 0.27, "ablation_dynamic_only": 0.29}
for name, tau in opt_t.items():
    p = np.load(os.path.join(res_dir, f"{name}_test_probs.npy"))
    m = compute_flat_metrics(p, y_dyn_flat, threshold=tau)
    pr_auc = average_precision_score(y_dyn_flat, p)
    roc_auc = roc_auc_score(y_dyn_flat, p)
    print(f"{name:24s} | IoU={m['iou']:.4f} | F1={m['f1']:.4f} | PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f}")
