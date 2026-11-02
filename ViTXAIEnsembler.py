import cv2
import numpy as np
import torch
import torch.nn.functional as F


class ViTXAIEnsembler:
    """
    Vision XAI Ensembler and Dual-Confidence Contour Extractor.
    Combines complementary vision attributions (e.g., Grad-CAM++ and Integrated Gradients)
    to generate robust, dual-margin tumor outlines (Core and Margin).
    """
    def __init__(self, xai_engine=None):
        self.xai = xai_engine

    def compute_ensembled_map(
        self,
        attr_map_1: np.ndarray,
        attr_map_2: np.ndarray,
        alpha: float = 0.50,
        fusion_mode: str = "harmonic"
    ) -> np.ndarray:
        """
        Fuses two vision attribution maps (e.g. Grad-CAM++ and Integrated Gradients).
        fusion_mode: 'harmonic' (strictest agreement), 'geometric', or 'weighted_linear'.
        """
        eps = 1e-6
        if fusion_mode == "harmonic":
            ensembled = (2.0 * attr_map_1 * attr_map_2) / (attr_map_1 + attr_map_2 + eps)
        elif fusion_mode == "geometric":
            ensembled = np.sqrt(attr_map_1 * attr_map_2 + eps)
        else:
            ensembled = alpha * attr_map_1 + (1.0 - alpha) * attr_map_2

        ensembled = (ensembled - ensembled.min()) / (ensembled.max() - ensembled.min() + eps)
        return ensembled

    def extract_dual_confidence_contours(
        self,
        base_img: np.ndarray,
        saliency_map: np.ndarray,
        core_multiplier: float = 1.35,
        margin_multiplier: float = 0.85,
        min_contour_area: int = 40
    ) -> dict:
        """
        Applies multi-level Otsu-anchored thresholding to extract:
        - High-confidence core tumor boundary (Red)
        - Extended peritumoral margin / architectural distortion zone (Cyan)
        """
        h, w = saliency_map.shape
        # Base canvas normalized for BGR display
        canvas = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Base threshold using Otsu's intra-class variance minimization
        scaled_map = (saliency_map * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(scaled_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Core Focal Boundary (Red: Higher confidence threshold)
        t_core = int(np.clip(otsu_val * core_multiplier, 0, 254))
        _, core_mask = cv2.threshold(scaled_map, t_core, 255, cv2.THRESH_BINARY)
        # Morphological opening to eliminate isolated speckles
        core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        raw_contours_core, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_core = [c for c in raw_contours_core if cv2.contourArea(c) >= min_contour_area]

        # 2. Extended Parenchymal Margin (Cyan: Lower threshold with closing filter)
        t_margin = int(np.clip(otsu_val * margin_multiplier, 0, 254))
        _, margin_mask = cv2.threshold(scaled_map, t_margin, 255, cv2.THRESH_BINARY)
        # Morphological closing to bridge spiculations and distortion folds
        margin_mask = cv2.morphologyEx(margin_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        raw_contours_margin, _ = cv2.findContours(margin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_margin = [c for c in raw_contours_margin if cv2.contourArea(c) >= min_contour_area]

        # 3. Render outlines onto canvas
        # Cyan: (B=255, G=255, R=0) -> Margin
        cv2.drawContours(canvas, contours_margin, -1, (255, 255, 0), 2)
        # Red: (B=0, G=0, R=255) -> Core
        cv2.drawContours(canvas, contours_core, -1, (0, 0, 255), 2)

        # Calculate bounding box coordinates and centroids for reporting
        core_rois = [cv2.boundingRect(c) for c in contours_core]
        margin_rois = [cv2.boundingRect(c) for c in contours_margin]

        return {
            "overlay_image": canvas,
            "core_mask": core_mask,
            "margin_mask": margin_mask,
            "core_contours": contours_core,
            "margin_contours": contours_margin,
            "core_boxes": core_rois,
            "margin_boxes": margin_rois,
            "thresholds": {"otsu": otsu_val, "core": t_core, "margin": t_margin}
        }