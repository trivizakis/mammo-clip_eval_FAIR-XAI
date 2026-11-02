import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def visualize_classification_xai(
    base_img: np.ndarray,
    grad_cam: np.ndarray,
    grad_cam_pp: np.ndarray,
    integrated_grads: np.ndarray,
    ensembled_map: np.ndarray,
    contour_overlay: np.ndarray,
    pred_prob: float,
    pred_decision: str,
    true_label: int,
    decision_threshold: float = 0.5000,
    save_path: str = None,
    dpi: int = 120
):
    """
    Renders an end-to-end 6-panel traditional classification XAI audit figure:
    1. Preprocessed Mammogram with Ground Truth & Prediction
    2. Grad-CAM Attribution Map
    3. Grad-CAM++ Fine-Grained Attribution Map
    4. Integrated Gradients Saliency Map
    5. Harmonic Ensembled Vision Agreement Map
    6. Dual Iso-Confidence Outlines (Red = Core, Cyan = Margin)
    """
    base_norm = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor="black")
    plt.subplots_adjust(wspace=0.06, hspace=0.14)

    true_str = "MALIGNANT" if true_label == 1 else "BENIGN"
    status_color = "#ff4444" if true_label == 1 else "#00e5ff"

    # Panel 1: Mammogram
    ax0 = axes[0, 0]
    ax0.set_facecolor("black")
    ax0.imshow(base_norm, cmap="gray")
    title_p1 = f"Preprocessed Mammogram [True: {true_str}]\nPredicted: {pred_decision} (Prob: {pred_prob:.4f} | Thresh: {decision_threshold:.4f})"
    ax0.set_title(title_p1, color="white", fontsize=11, pad=8, weight="bold")
    ax0.text(0.5, -0.05, f"Image-level Classification | Ground Truth: {true_str}", transform=ax0.transAxes, color=status_color, fontsize=9.5, ha="center")
    ax0.axis("off")

    # Panel 2: Grad-CAM
    ax1 = axes[0, 1]
    ax1.set_facecolor("black")
    im1 = ax1.imshow(grad_cam, cmap="magma")
    im1.set_clim(0, 1)
    ax1.set_title("Grad-CAM Attribution Map\n(Penultimate C5 Coarse Localization)", color="white", fontsize=11, pad=8, weight="bold")
    ax1.axis("off")

    # Panel 3: Grad-CAM++
    ax2 = axes[0, 2]
    ax2.set_facecolor("black")
    im2 = ax2.imshow(grad_cam_pp, cmap="inferno")
    im2.set_clim(0, 1)
    ax2.set_title("Grad-CAM++ Attribution Map\n(Higher-Order Focal Lesion Weighting)", color="white", fontsize=11, pad=8, weight="bold")
    ax2.axis("off")

    # Panel 4: Integrated Gradients
    ax3 = axes[1, 0]
    ax3.set_facecolor("black")
    im3 = ax3.imshow(integrated_grads, cmap="viridis")
    im3.set_clim(0, 1)
    ax3.set_title("Integrated Gradients Saliency Map\n(Axiomatic Path Attribution)", color="white", fontsize=11, pad=8, weight="bold")
    ax3.axis("off")

    # Panel 5: Harmonic Ensembled Map
    ax4 = axes[1, 1]
    ax4.set_facecolor("black")
    im4 = ax4.imshow(ensembled_map, cmap="plasma")
    im4.set_clim(0, 1)
    ax4.set_title("Harmonic Ensembled Vision Map\n(Multi-Method Agreement Fusion)", color="white", fontsize=11, pad=8, weight="bold")
    ax4.axis("off")

    # Panel 6: Dual Iso-Confidence Outlines
    ax5 = axes[1, 2]
    ax5.set_facecolor("black")
    if contour_overlay is not None and contour_overlay.ndim == 3:
        ax5.imshow(contour_overlay)
    else:
        ax5.imshow(base_norm, cmap="gray")
    ax5.set_title("Dual Iso-Confidence Outlines\n(Red: Core Tumor | Cyan: Margin)", color="white", fontsize=11, pad=8, weight="bold")
    ax5.text(0.5, -0.05, "Red = High-Confidence Core | Cyan = Peritumoral Margin", transform=ax5.transAxes, color="#00ffff", fontsize=9.5, ha="center")
    ax5.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
