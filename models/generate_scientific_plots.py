import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Use a clean, professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 15,
    'legend.fontsize': 11,
    'grid.alpha': 0.4,
    'font.family': 'sans-serif'
})

def compute_ece(probs, y_true, n_bins=10):
    """
    Expected Calibration Error (ECE) implementation.
    """
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    
    for i in range(n_bins):
        bin_idx = (probs > bin_edges[i]) & (probs <= bin_edges[i+1])
        bin_weight = np.sum(bin_idx) / n_samples
        if bin_weight > 0:
            bin_pred = probs[bin_idx]
            bin_true = y_true[bin_idx]
            bin_score = np.abs(np.mean(bin_true) - np.mean(bin_pred))
            ece += bin_weight * bin_score
    return ece

def main():
    print("=================================================================")
    print("--- Generating Scientific Plots (PR Curves & Calibration) ---")
    print("=================================================================")
    
    data_dir = "engineering_validation_results"
    
    # Load test prediction arrays
    models_data = {
        "MFPIT": {
            "probs": np.load(os.path.join(data_dir, "mfpit_test_probs.npy")),
            "targets": np.load(os.path.join(data_dir, "mfpit_test_targets.npy")),
            "color": "#1f77b4", # cobalt blue
            "linestyle": "-"
        },
        "GBDT": {
            "probs": np.load(os.path.join(data_dir, "gbdt_test_probs.npy")),
            "targets": np.load(os.path.join(data_dir, "gbdt_test_targets.npy")),
            "color": "#2ca02c", # emerald green
            "linestyle": "--"
        },
        "Random Forest": {
            "probs": np.load(os.path.join(data_dir, "random_forest_test_probs.npy")),
            "targets": np.load(os.path.join(data_dir, "random_forest_test_targets.npy")),
            "color": "#ff7f0e", # vibrant orange
            "linestyle": "-."
        }
    }
    
    # Output paths
    pr_out = "../research_final_for_prism/pr_curves.png"
    cal_out = "../research_final_for_prism/calibration_curves.png"
    
    # Ensure research dir exists
    os.makedirs("../research_final_for_prism", exist_ok=True)
    
    # -----------------------------------------------------------------
    # Plot 1: Precision-Recall Curves
    # -----------------------------------------------------------------
    print("Generating Precision-Recall Curve Plot...")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    for name, cfg in models_data.items():
        probs = cfg["probs"]
        targets = cfg["targets"]
        
        # Calculate PR curve and AUC
        precision, recall, _ = precision_recall_curve(targets, probs)
        pr_auc = average_precision_score(targets, probs)
        
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", 
                color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2.0)
        
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall (PR) Curves on Temporal Holdout")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", shadow=False)
    plt.tight_layout()
    plt.savefig(pr_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Saved PR Curves to {pr_out}")
    
    # -----------------------------------------------------------------
    # Plot 2: Calibration Reliability Diagrams
    # -----------------------------------------------------------------
    print("Generating Calibration Reliability Diagrams...")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Perfectly calibrated reference line
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="Perfect Calibration", linewidth=1.5)
    
    for name, cfg in models_data.items():
        probs = cfg["probs"]
        targets = cfg["targets"]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(targets, probs, n_bins=10, strategy='uniform')
        ece = compute_ece(probs, targets, n_bins=10)
        brier = brier_score_loss(targets, probs)
        
        ax.plot(prob_pred, prob_true, marker='o', label=f"{name} (ECE = {ece:.4f}, Brier = {brier:.4f})",
                color=cfg["color"], linestyle=cfg["linestyle"], linewidth=1.8, markersize=5)
        
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Empirical)")
    ax.set_title("Calibration Reliability Diagrams (ECE & Brier)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=False)
    plt.tight_layout()
    plt.savefig(cal_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Saved Calibration Reliability Diagrams to {cal_out}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
