import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_threshold_sensitivity_analysis(y_true: np.ndarray, y_probs: np.ndarray, beta: float = 1.5, save_plot_path: str = None) -> dict:
    """
    Calibrates classification probability decision thresholds on validation data,
    optimizing for the maximum overall Sensitivity at the best Specificity balance,
    prints sensitivity and operating characteristics across thresholds,
    and plots the sensitivity curve.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    # Dense sweep to find the Calibrated Optimal Threshold (Max Sensitivity @ Best Specificity Balance)
    dense_thresholds = np.linspace(0.01, 0.99, 300)
    best_sn_sp_balance = -1.0
    optimal_threshold = 0.5

    sn_curve = []
    sp_curve = []
    bacc_curve = []
    harmonic_bal_curve = []

    for t in dense_thresholds:
        preds = (y_probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        sn = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        bacc = (sn + sp) / 2.0
        h_bal = 2.0 * sn * sp / (sn + sp + 1e-8)

        sn_curve.append(sn)
        sp_curve.append(sp)
        bacc_curve.append(bacc)
        harmonic_bal_curve.append(h_bal)

        # Maximize Sensitivity at the best Specificity balance
        custom_balance = (sn * 1.25 + sp * 0.75) if (sn >= 0.65 and sp >= 0.20) else bacc
        if custom_balance > best_sn_sp_balance:
            best_sn_sp_balance = custom_balance
            optimal_threshold = t

    # Formatted sensitivity curve table across standard probability thresholds
    display_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    table_rows = []
    print("\n" + "=" * 80)
    print("      VALIDATION SET SENSITIVITY CURVE & OPTIMAL THRESHOLD CALIBRATION")
    print("=" * 80)
    print(f"{'Threshold':^11} | {'Sensitivity (SN)':^18} | {'Specificity (SP)':^18} | {'Balanced ACC':^14} | {'Harmonic Bal':^12}")
    print("-" * 80)

    for dt in display_thresholds:
        preds = (y_probs >= dt).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        sn = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        bacc = (sn + sp) / 2.0
        h_bal = 2.0 * sn * sp / (sn + sp + 1e-8)

        table_rows.append({
            "threshold": dt,
            "sensitivity": round(float(sn), 4),
            "specificity": round(float(sp), 4),
            "balanced_accuracy": round(float(bacc), 4),
            "harmonic_balance": round(float(h_bal), 4)
        })
        print(f"{dt:^11.2f} | {sn:^18.4f} | {sp:^18.4f} | {bacc:^14.4f} | {h_bal:^12.4f}")
    print("=" * 80)

    # Plot sensitivity curve if requested
    if save_plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(dense_thresholds, sn_curve, label="Sensitivity (Recall / SN)", color="#d62728", lw=2.2)
        ax.plot(dense_thresholds, sp_curve, label="Specificity (SP)", color="#1f77b4", lw=2.0, ls="--")
        ax.plot(dense_thresholds, bacc_curve, label="Balanced Accuracy", color="#2ca02c", lw=2.0, ls=":")
        ax.plot(dense_thresholds, harmonic_bal_curve, label="Harmonic SN-SP Balance", color="#8c564b", lw=1.8, ls="-.")

        ax.axvline(optimal_threshold, color="#ff7f0e", linestyle="-", lw=2.4,
                   label=f"Calibrated Optimal Threshold ({optimal_threshold:.3f})")

        ax.set_title("Mammography Validation Sensitivity-Specificity Calibration", fontsize=13, fontweight="bold")
        ax.set_xlabel("Classification Probability Decision Threshold", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9.5)
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close(fig)
        print(f"[*] Sensitivity curve plot saved to -> {save_plot_path}")

    return {
        "calibrated_optimal_threshold": float(optimal_threshold),
        "optimal_sn_sp_balance_threshold": float(optimal_threshold),
        "table_rows": table_rows
    }