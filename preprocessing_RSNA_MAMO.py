import os
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_and_window_dicom(dcm_or_path) -> np.ndarray:
    """
    VOI-LUT Windowing & MONOCHROME Handling:
    Applies Value of Interest Look-Up Table (VOI-LUT) windowing and properly handles
    MONOCHROME1 / MONOCHROME2 photometric interpretations to ensure consistent pixel intensity scaling.
    """
    if isinstance(dcm_or_path, str):
        dcm = pydicom.dcmread(dcm_or_path)
    else:
        dcm = dcm_or_path

    raw = dcm.pixel_array.astype(np.float32)

    # 1. Apply VOI-LUT windowing if present
    try:
        raw = apply_voi_lut(raw, dcm)
    except Exception:
        pass

    # 2. Handle Photometric Interpretation (MONOCHROME1 -> MONOCHROME2 inversion)
    # In MONOCHROME1, air is white (high intensity) and dense tissue is dark (low intensity).
    # Inverting MONOCHROME1 ensures consistent representation where dense tissue is bright.
    pi = getattr(dcm, "PhotometricInterpretation", "")
    if str(pi).strip().upper() == "MONOCHROME1":
        raw = np.amax(raw) - raw

    # 3. Min-Max normalization to [0, 1] range
    denom = raw.max() - raw.min()
    if denom > 1e-6:
        norm = (raw - raw.min()) / denom
    else:
        norm = np.zeros_like(raw, dtype=np.float32)

    return np.ascontiguousarray(norm, dtype=np.float32)


class MammogramPreprocessor:
    """
    Performs VOI-LUT Windowing & MONOCHROME standardization, unsupervised body mask extraction,
    anatomical pectoral muscle excision (applied after VOI-LUT/MONOCHROME handling for MLO views),
    site-specific piecewise linear intensity transformation, and standardization.
    """
    def __init__(self, target_size=(1536, 1024), site_stats=None):
        self.target_size = target_size
        # site_stats maps site_id -> {'p_low': float, 'p_mid': float, 'p_high': float, 'mean': float, 'std': float}
        self.site_stats = site_stats or {
            1: {'p_low': 0.05, 'p_mid': 0.50, 'p_high': 0.95, 'mean': 0.28, 'std': 0.22},
            2: {'p_low': 0.02, 'p_mid': 0.45, 'p_high': 0.98, 'mean': 0.31, 'std': 0.25}
        }

    def segment_breast_tissue(self, norm_img: np.ndarray, laterality: str = 'R', view: str = 'MLO') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts body mask and applies anatomical pectoral muscle removal.
        - Preserves Craniocaudal (CC) views where pectoral muscle is anatomically absent.
        - For Mediolateral Oblique (MLO) views, excises the triangular pectoral muscle anchored
          at the superior-posterior (top-left) chest wall border.
        """
        norm = norm_img.copy()

        # Standardize breast orientation (ensure chest wall is on the Left border)
        flipped = False
        if str(laterality).strip().upper() == 'R':
            norm = np.fliplr(norm)
            flipped = True

        # Whole breast mask extraction (ensuring no holes, analyzing whole breast including fatty adipose tissue)
        blur = cv2.GaussianBlur((norm * 255).astype(np.uint8), (7, 7), 0)
        otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Capture low-density adipose tissue down to skin boundary
        low_thresh = max(5, int(otsu_val * 0.25))
        thresh = (blur >= low_thresh).astype(np.uint8)

        # Morphological closing to bridge boundary gaps and connect tissue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Keep only the largest connected component (the breast, eliminating markers and tags)
        labels, num_features = ndimage.label(closed)
        if num_features > 0:
            sizes = ndimage.sum(closed, labels, range(num_features + 1))
            largest_label = np.argmax(sizes)
            breast_mask = (labels == largest_label).astype(np.uint8)
            # Fill all internal holes completely
            breast_mask = ndimage.binary_fill_holes(breast_mask).astype(np.uint8)
        else:
            breast_mask = np.ones_like(thresh, dtype=np.uint8)

        # Anatomical pectoral muscle segmentation
        # CC views anatomically do not feature the pectoral muscle; attempting removal erodes healthy glandular parenchyma.
        is_mlo = str(view).strip().upper() == 'MLO'
        muscle_mask = np.zeros_like(breast_mask, dtype=np.uint8)

        if is_mlo:
            h, w = breast_mask.shape
            quad_h = int(h * 0.40)
            quad_w = int(w * 0.45)

            roi = norm[:quad_h, :quad_w].copy()
            roi_mask = breast_mask[:quad_h, :quad_w]
            roi[roi_mask == 0] = 0

            valid_pixels = roi[roi_mask > 0]
            if len(valid_pixels) > 200:
                roi_u8 = (roi * 255).astype(np.uint8)
                t_val, _ = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                dense = ((roi_u8 >= t_val) & (roi_mask > 0)).astype(np.uint8)

                # Find connected components in dense tissue
                lbls, num_f = ndimage.label(dense)
                if num_f > 0:
                    # The pectoral muscle is anchored at the superior and lateral edges (top and left borders)
                    top_margin = lbls[0, :int(quad_w * 0.6)]
                    left_margin = lbls[:int(quad_h * 0.6), 0]
                    corner_candidates = set(top_margin).intersection(set(left_margin))
                    corner_candidates.discard(0)

                    if not corner_candidates:
                        # Fallback: check candidates touching top margin or left margin near corner
                        near_corner = set(top_margin[:int(quad_w * 0.3)]).union(set(left_margin[:int(quad_h * 0.3)]))
                        near_corner.discard(0)
                        corner_candidates = near_corner

                    if corner_candidates:
                        # Select the largest corner-anchored component
                        target_lbl = max(corner_candidates, key=lambda l: np.sum(lbls == l))
                        muscle_comp = (lbls == target_lbl).astype(np.uint8)

                        # Form the smooth convex triangular contour of the pectoral muscle
                        contours, _ = cv2.findContours(muscle_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            hull = cv2.convexHull(contours[0])
                            hull_mask = np.zeros_like(muscle_comp)
                            cv2.drawContours(hull_mask, [hull], -1, 1, -1)
                            # Ensure muscle mask is constrained within breast tissue
                            muscle_mask[:quad_h, :quad_w] = (hull_mask * roi_mask).astype(np.uint8)

        # Whole breast tissue extraction (preserving entire breast with zero internal holes)
        final_mask = (breast_mask & (~muscle_mask)).astype(np.uint8)
        final_mask = ndimage.binary_fill_holes(final_mask).astype(np.uint8)
        segmented_tissue = norm * final_mask

        # Restore original lateral coordinate space if flipped
        if flipped:
            norm = np.ascontiguousarray(np.fliplr(norm))
            breast_mask = np.ascontiguousarray(np.fliplr(breast_mask))
            muscle_mask = np.ascontiguousarray(np.fliplr(muscle_mask))
            final_mask = np.ascontiguousarray(np.fliplr(final_mask))
            segmented_tissue = np.ascontiguousarray(np.fliplr(segmented_tissue))
        else:
            norm = np.ascontiguousarray(norm)
            breast_mask = np.ascontiguousarray(breast_mask)
            muscle_mask = np.ascontiguousarray(muscle_mask)
            final_mask = np.ascontiguousarray(final_mask)
            segmented_tissue = np.ascontiguousarray(segmented_tissue)

        return segmented_tissue, final_mask, breast_mask, muscle_mask

    def preprocess_full(self, dcm_or_img, laterality: str = 'R', view: str = 'MLO', site_id: int = 1) -> dict:
        """
        Runs full preprocessing pipeline:
        1. VOI-LUT Windowing & MONOCHROME standardization
        2. Breast mask segmentation & anatomical pectoral excision
        3. Piecewise linear contrast expansion & clinical standardization
        """
        if isinstance(dcm_or_img, (str, pydicom.dataset.FileDataset)):
            norm = read_and_window_dicom(dcm_or_img)
        else:
            img_arr = np.asarray(dcm_or_img, dtype=np.float32)
            denom = img_arr.max() - img_arr.min()
            norm = (img_arr - img_arr.min()) / (denom + 1e-8) if denom > 1e-6 else np.zeros_like(img_arr)

        segmented_tissue, final_mask, breast_mask, muscle_mask = self.segment_breast_tissue(
            norm, laterality=laterality, view=view
        )

        preprocessed_img = self.piecewise_linear_transform(segmented_tissue, final_mask, site_id=site_id)

        return {
            "raw_normalized": norm,
            "breast_mask": breast_mask,
            "muscle_mask": muscle_mask,
            "final_mask": final_mask,
            "segmented_tissue": segmented_tissue,
            "preprocessed_img": preprocessed_img
        }

    def piecewise_linear_transform(self, img: np.ndarray, mask: np.ndarray, site_id: int) -> np.ndarray:
        valid_pixels = img[mask > 0]
        if len(valid_pixels) == 0:
            res = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            return np.ascontiguousarray(res, dtype=np.float32)

        # Dynamic contrast landmarks from tissue histogram
        cfg = self.site_stats.get(int(site_id), self.site_stats[1])
        q1 = np.quantile(valid_pixels, cfg['p_low'])
        q2 = np.quantile(valid_pixels, cfg['p_mid'])
        q3 = np.quantile(valid_pixels, cfg['p_high'])

        transformed = np.zeros_like(img, dtype=np.float32)

        # Piecewise 3-zone contrast expansion: background suppression, parenchyma expansion, calcification preservation
        idx1 = (img <= q1) & (mask > 0)
        idx2 = (img > q1) & (img <= q2) & (mask > 0)
        idx3 = (img > q2) & (img <= q3) & (mask > 0)
        idx4 = (img > q3) & (mask > 0)

        transformed[idx1] = 0.15 * ((img[idx1] - img.min()) / (q1 - img.min() + 1e-6))
        transformed[idx2] = 0.15 + 0.35 * ((img[idx2] - q1) / (q2 - q1 + 1e-6))
        transformed[idx3] = 0.50 + 0.35 * ((img[idx3] - q2) / (q3 - q2 + 1e-6))
        transformed[idx4] = 0.85 + 0.15 * ((img[idx4] - q3) / (img.max() - q3 + 1e-6))

        # Site-specific standardization (zero-mean and unit-variance) on segmented breast tissue
        cfg = self.site_stats.get(int(site_id), self.site_stats[1])
        site_mean = float(cfg.get('mean', 0.4836))
        site_std = float(cfg.get('std', 0.2012))
        transformed[mask > 0] = (transformed[mask > 0] - site_mean) / (site_std + 1e-6)
        transformed[mask == 0] = 0.0

        res = cv2.resize(transformed, (self.target_size[1], self.target_size[0]))
        return np.ascontiguousarray(res, dtype=np.float32)


def save_preprocessing_steps(patient_id: str, image_id: str, steps: dict, output_base_dir: str) -> str:
    """
    Saves each individual preprocessing stage and an overview diagnostic progression panel
    under output_base_dir/{patient_id}_{image_id}/.
    """
    case_dir = os.path.join(output_base_dir, f"{patient_id}_{image_id}")
    os.makedirs(case_dir, exist_ok=True)

    # 1. Step 1: VOI-LUT Windowing & MONOCHROME Standardization
    raw_u8 = (np.clip(steps["raw_normalized"], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(case_dir, "step1_voi_lut_monochrome.png"), raw_u8)
    cv2.imwrite(os.path.join(case_dir, "step1_raw_normalized.png"), raw_u8)

    # 2. Step 2: Unsupervised Body Mask
    mask_u8 = (steps["breast_mask"] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(case_dir, "step2_breast_mask.png"), mask_u8)

    # 3. Step 3: Anatomical Pectoral Muscle Mask
    muscle_u8 = (steps["muscle_mask"] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(case_dir, "step3_pectoral_muscle_mask.png"), muscle_u8)

    # 4. Step 4: Segmented Parenchyma Tissue
    tissue_u8 = (np.clip(steps["segmented_tissue"], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(case_dir, "step4_segmented_tissue.png"), tissue_u8)

    # 5. Step 5: Final Standardized Input Image
    prep = steps["preprocessed_img"]
    prep_norm = (prep - prep.min()) / (prep.max() - prep.min() + 1e-8)
    prep_u8 = (prep_norm * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(case_dir, "step5_preprocessed_standardized.png"), prep_u8)

    # Save exact tensor array as npy
    np.save(os.path.join(case_dir, "preprocessed_img.npy"), prep)

    # 6. Fast, thread-safe, high-resolution 5-stage progression panel via OpenCV
    thumb_h, thumb_w = 512, 340
    p1 = cv2.resize(cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))
    p2 = cv2.resize(cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))

    p3_base = cv2.resize(cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))
    m_res = cv2.resize(muscle_u8, (thumb_w, thumb_h))
    p3_base[m_res > 0] = [0, 0, 255]  # Red pectoral highlight
    p3 = p3_base

    p4 = cv2.resize(cv2.cvtColor(tissue_u8, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))
    prep_color = cv2.applyColorMap(prep_u8, cv2.COLORMAP_INFERNO)
    p5 = cv2.resize(prep_color, (thumb_w, thumb_h))

    header_h = 42
    panel_imgs = []
    titles = ["1. VOI-LUT Windowed", "2. Body Mask", "3. Pectoral Excision", "4. Parenchyma", "5. Standardized Input"]
    for p_img, title in zip([p1, p2, p3, p4, p5], titles):
        header = np.zeros((header_h, thumb_w, 3), dtype=np.uint8)
        cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        panel_imgs.append(np.vstack([header, p_img]))

    composite = np.hstack(panel_imgs)
    summary_path = os.path.join(case_dir, "preprocessing_progression_panel.png")
    cv2.imwrite(summary_path, composite)

    return case_dir