import cv2
import numpy as np

def draw_probable_tumor_boundaries(base_img: np.ndarray, xai_map: np.ndarray) -> np.ndarray:
    # Convert single-channel base image to BGR for clinical overlay
    canvas = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    scaled_map = (xai_map * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(scaled_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. High Confidence Core (Red Outline)
    core_thresh = int(np.clip(otsu_thresh * 1.35, 0, 254))
    _, core_mask = cv2.threshold(scaled_map, core_thresh, 255, cv2.THRESH_BINARY)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours_core, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Extended Peritumoral / Margin Zone (Cyan Outline)
    margin_thresh = int(np.clip(otsu_thresh * 0.85, 0, 254))
    _, margin_mask = cv2.threshold(scaled_map, margin_thresh, 255, cv2.THRESH_BINARY)
    margin_mask = cv2.morphologyEx(margin_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours_margin, _ = cv2.findContours(margin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Render contours directly onto canvas
    cv2.drawContours(canvas, contours_margin, -1, (255, 255, 0), 2) # Cyan = Extended margin
    cv2.drawContours(canvas, contours_core, -1, (0, 0, 255), 2)     # Red = Core focal boundary
    
    return canvas