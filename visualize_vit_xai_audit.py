import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize_vit_xai_audit(
    base_img: np.ndarray,
    attr_map: np.ndarray,
    text_map: np.ndarray,
    ensembled_map: np.ndarray,
    contour_results: dict,
    save_path: str = None,
    dpi: int = 150
):
    """
    Renders an end-to-end 6-panel vision explainability diagnostic audit:
    1. Preprocessed Mammogram
    2. Grad-CAM++ Attribution Map
    3. Integrated Gradients Saliency Map
    4. Harmonic Ensembled Vision Map (Dual-Vision Agreement)
    5. Alpha-Blended Tissue Activation
    6. Dual Iso-Confidence Outlines (Red=Core Tumor, Cyan=Peritumoral Margin)
    """
    # 1. Prepare base image in [0, 1] range
    base_norm = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    base_uint8 = (base_norm * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2RGB)

    # 2. Generate alpha-blended heatmap overlay
    heatmap_colored = cv2.applyColorMap((ensembled_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Restrict heatmap overlay to non-zero breast tissue regions
    tissue_mask = (base_uint8 > 5)[:, :, None]
    blended_overlay = np.where(
        tissue_mask,
        cv2.addWeighted(base_rgb, 0.55, heatmap_colored, 0.45, 0),
        base_rgb
    )

    # 3. Retrieve contour overlay from ensembler results
    contour_overlay_rgb = cv2.cvtColor(contour_results["overlay_image"], cv2.COLOR_BGR2RGB)

    # 4. Multi-panel plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 14), facecolor="black")
    plt.subplots_adjust(wspace=0.04, hspace=0.12)

    panels = [
        (base_norm, "Preprocessed Mammogram", "gray", None),
        (attr_map, "Grad-CAM++ Attribution Map", "magma", (0, 1)),
        (text_map, "Integrated Gradients Saliency Map", "viridis", (0, 1)),
        (ensembled_map, "Harmonic Ensembled Map (Vision Agreement)", "inferno", (0, 1)),
        (blended_overlay, "Alpha-Blended Tissue Activation", None, None),
        (contour_overlay_rgb, "Dual Iso-Confidence Outlines\n(Red=Core, Cyan=Margin)", None, None)
    ]

    for ax, (data, title, cmap, clim) in zip(axes.flat, panels):
        ax.set_facecolor("black")
        if cmap:
            im = ax.imshow(data, cmap=cmap)
            if clim:
                im.set_clim(clim[0], clim[1])
        else:
            ax.imshow(data)

        ax.set_title(title, color="white", fontsize=11, pad=8, weight="bold")
        ax.axis("off")

    # Add threshold metadata annotation on the contour panel
    thresh_info = contour_results.get("thresholds", {})
    t_text = f"Otsu: {thresh_info.get('otsu', 0)} | Core: {thresh_info.get('core', 0)} | Margin: {thresh_info.get('margin', 0)}"
    axes[1, 2].text(
        0.5, -0.05, t_text,
        transform=axes[1, 2].transAxes,
        color="#00ffff", fontsize=10, ha="center"
    )

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)