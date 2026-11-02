import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_detector_xai(
    base_img: np.ndarray,
    det_cam: np.ndarray,
    cls_saliency: np.ndarray,
    loc_saliency: np.ndarray,
    drise_map: np.ndarray,
    factor_map: np.ndarray,
    mc_variability: np.ndarray,
    bbox_info: dict,
    true_label: int = None,
    save_path: str = None,
    dpi: int = 150
):
    """
    Renders an end-to-end 6-panel diagnostic audit figure for the Mammo-CLIP detector:
    1. Preprocessed Mammogram with Predicted Bounding Box and Ground Truth Label
    2. Det-CAM Attribution Map (FPN multi-scale proposal evidence)
    3. Separated Localization vs. Classification Saliency (Classification heatmap + Localization contours)
    4. D-RISE Detector Saliency (Randomized input sampling verification)
    5. Mammo-FActOR Concept Saliency (Dense CLIP cosine similarity to 'Mass')
    6. Monte Carlo Prediction Variability (Spatial prediction uncertainty map)
    """
    # 1. Normalize base image to [0, 1]
    base_norm = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    base_uint8 = (base_norm * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2RGB)

    H, W = base_img.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor="black")
    plt.subplots_adjust(wspace=0.06, hspace=0.14)

    # -------------------------------------------------------------
    # Panel 1: Mammogram with Predicted Bounding Box
    # -------------------------------------------------------------
    ax0 = axes[0, 0]
    ax0.set_facecolor("black")
    ax0.imshow(base_norm, cmap="gray")

    bbox = bbox_info.get("bbox", [0, 0, W, H])
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    is_malignant = bbox_info.get("is_malignant", False)
    score = bbox_info.get("score", 0.0)
    opt_thresh = bbox_info.get("opt_threshold", 0.0)
    decision = bbox_info.get("decision", "BENIGN")

    box_color = "#ff2222" if is_malignant else "#00e5ff"
    rect = patches.Rectangle(
        (x1, y1), bw, bh,
        linewidth=2.5, edgecolor=box_color, facecolor="none", linestyle="-"
    )
    ax0.add_patch(rect)

    true_str = f" [True: {'MALIGNANT' if true_label == 1 else 'BENIGN'}]" if true_label is not None else ""
    title_p1 = f"Preprocessed Mammogram{true_str}\nPredicted Box: {decision} (Score: {score:.4f} | Thresh: {opt_thresh:.4f})"
    ax0.set_title(title_p1, color="white", fontsize=11, pad=8, weight="bold")
    ax0.text(
        0.5, -0.06, f"BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] | Area: {bw*bh:.0f} px²",
        transform=ax0.transAxes, color=box_color, fontsize=9.5, ha="center"
    )
    ax0.axis("off")

    # -------------------------------------------------------------
    # Panel 2: Det-CAM Attribution Map
    # -------------------------------------------------------------
    ax1 = axes[0, 1]
    ax1.set_facecolor("black")
    im1 = ax1.imshow(det_cam, cmap="inferno")
    im1.set_clim(0, 1)
    ax1.set_title("Det-CAM Attribution Map\n(FPN Anchor Proposal Evidence)", color="white", fontsize=11, pad=8, weight="bold")
    ax1.axis("off")

    # -------------------------------------------------------------
    # Panel 3: Separated Localization vs. Classification Saliency
    # -------------------------------------------------------------
    ax2 = axes[0, 2]
    ax2.set_facecolor("black")
    im2 = ax2.imshow(cls_saliency, cmap="hot")
    im2.set_clim(0, 1)

    # Overlay Localization Saliency as cyan contour lines
    loc_norm = (loc_saliency * 255).astype(np.uint8)
    contours_loc, _ = cv2.findContours(
        cv2.threshold(loc_norm, 120, 255, cv2.THRESH_BINARY)[1],
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours_loc:
        if len(c) > 3:
            pts = c.squeeze()
            if pts.ndim == 2:
                ax2.plot(pts[:, 0], pts[:, 1], color="#00ffff", linewidth=1.5, linestyle="--")

    ax2.set_title("Separated Saliency\n(Heatmap: Class | Cyan Contours: Localization)", color="white", fontsize=11, pad=8, weight="bold")
    ax2.axis("off")

    # -------------------------------------------------------------
    # Panel 4: D-RISE Detector Saliency
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("black")
    im3 = ax3.imshow(drise_map, cmap="viridis")
    im3.set_clim(0, 1)
    ax3.set_title("D-RISE Saliency Map\n(Randomized Mask Perturbation)", color="white", fontsize=11, pad=8, weight="bold")
    ax3.axis("off")

    # -------------------------------------------------------------
    # Panel 5: Mammo-FActOR Concept Saliency
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("black")
    im4 = ax4.imshow(factor_map, cmap="plasma")
    im4.set_clim(0, 1)
    ax4.set_title("Mammo-FActOR Saliency Map\n(Dense CLIP Alignment to Concept 'Mass')", color="white", fontsize=11, pad=8, weight="bold")
    ax4.axis("off")

    # -------------------------------------------------------------
    # Panel 6: Monte Carlo Prediction Variability
    # -------------------------------------------------------------
    ax5 = axes[1, 2]
    ax5.set_facecolor("black")
    im5 = ax5.imshow(mc_variability, cmap="coolwarm")
    im5.set_clim(0, 1)
    score_std = bbox_info.get("score_std", 0.0)
    ax5.set_title("Monte Carlo Prediction Variability\n(Spatial Uncertainty / Prediction Stability)", color="white", fontsize=11, pad=8, weight="bold")
    ax5.text(
        0.5, -0.06, f"Pixel Variance Map σ²(x, y) | Score Std: {score_std:.4f}",
        transform=ax5.transAxes, color="#ffaa00", fontsize=9.5, ha="center"
    )
    ax5.axis("off")

    # Save artifact
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
