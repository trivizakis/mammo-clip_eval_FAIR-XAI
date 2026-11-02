import cv2
import numpy as np
import pandas as pd

class BIRADSShapeDescriptorExtractor:
    """
    Computes quantitative morphological and boundary descriptors
    from OpenCV contour coordinates for clinical BI-RADS categorization.
    """
    def __init__(self, pixel_spacing_mm: float = 0.1):
        """
        pixel_spacing_mm: Physical pixel spacing (e.g., 0.1 mm/pixel or 0.07 mm/pixel)
                          from DICOM tag (0028, 0030) to compute physical metric areas.
        """
        self.pixel_spacing = pixel_spacing_mm

    def extract_contour_metrics(self, contour: np.ndarray, roi_label: str = "Core_Nidus", roi_idx: int = 0) -> dict:
        pts = contour.squeeze()
        if len(pts.shape) != 2 or len(pts) < 5:
            return None

        # 1. Fundamental Geometric Measurements
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, closed=True)
        if area_px <= 0 or perimeter_px <= 0:
            return None

        area_mm2 = area_px * (self.pixel_spacing ** 2)
        perimeter_mm = perimeter_px * self.pixel_spacing

        # 2. Convex Hull & Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, closed=True)

        solidity = float(area_px / (hull_area + 1e-8))
        convexity = float(hull_perimeter / (perimeter_px + 1e-8))

        # 3. Compactness & Circularity (Isoperimetric Quotient)
        circularity = float((4.0 * np.pi * area_px) / (perimeter_px ** 2 + 1e-8))
        compactness = float((perimeter_px ** 2) / (area_px + 1e-8))

        # 4. Spiculation Index
        spiculation_index = float(np.clip((perimeter_px - hull_perimeter) / (perimeter_px + 1e-8), 0.0, 1.0))

        # 5. Inertia Ellipse (Moments & Aspect Ratio)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

        # Fit ellipse (requires >= 5 contour vertices)
        try:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (axis_minor, axis_major), orientation_angle = ellipse
        except Exception:
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (axis_minor, axis_major), orientation_angle = rect

        major_axis_len = max(axis_major, axis_minor)
        minor_axis_len = min(axis_major, axis_minor)
        aspect_ratio = float(major_axis_len / (minor_axis_len + 1e-8))

        semi_major = major_axis_len / 2.0
        semi_minor = minor_axis_len / 2.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (semi_minor ** 2 / (semi_major ** 2 + 1e-8)))))

        # 6. Radial Distance Distribution & Normalized Radial Variance (NRV)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        radial_distances = np.sqrt(dx ** 2 + dy ** 2)
        mean_r = np.mean(radial_distances)
        std_r = np.std(radial_distances)
        nrv = float(std_r / (mean_r + 1e-8))
        roughness_ratio = float((np.max(radial_distances) - np.min(radial_distances)) / (mean_r + 1e-8))

        # 7. BI-RADS Rule-Based Heuristic Estimation
        shape_category = self._classify_birads_shape(circularity, aspect_ratio, solidity)
        margin_category = self._classify_birads_margin(spiculation_index, nrv, convexity)

        return {
            "roi_id": f"{roi_label}_{roi_idx}",
            "roi_label": roi_label,
            "centroid_x_px": round(cx, 2),
            "centroid_y_px": round(cy, 2),
            "area_mm2": round(area_mm2, 2),
            "perimeter_mm": round(perimeter_mm, 2),
            "circularity": round(circularity, 4),
            "solidity": round(solidity, 4),
            "spiculation_index": round(spiculation_index, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "eccentricity": round(eccentricity, 4),
            "normalized_radial_variance": round(nrv, 4),
            "roughness_ratio": round(roughness_ratio, 4),
            "predicted_birads_shape": shape_category,
            "predicted_birads_margin": margin_category
        }

    @staticmethod
    def _classify_birads_shape(circularity: float, aspect_ratio: float, solidity: float) -> str:
        if circularity >= 0.70 and aspect_ratio < 1.35 and solidity >= 0.92:
            return "Round"
        elif aspect_ratio >= 1.35 and solidity >= 0.88:
            return "Oval"
        else:
            return "Irregular"

    @staticmethod
    def _classify_birads_margin(spiculation_index: float, nrv: float, convexity: float) -> str:
        if spiculation_index >= 0.20 or nrv >= 0.35:
            return "Spiculated"
        elif convexity < 0.80 or nrv >= 0.22:
            return "Microlobulated / Indistinct"
        else:
            return "Circumscribed"

    def process_all_contours(self, contour_results: dict) -> pd.DataFrame:
        """
        Parses all extracted core and peritumoral margin contours into a structured DataFrame.
        """
        records = []
        for idx, c in enumerate(contour_results.get("core_contours", [])):
            metrics = self.extract_contour_metrics(c, roi_label="Core_Nidus", roi_idx=idx)
            if metrics:
                records.append(metrics)

        for idx, c in enumerate(contour_results.get("margin_contours", [])):
            metrics = self.extract_contour_metrics(c, roi_label="Peritumoral_Margin", roi_idx=idx)
            if metrics:
                records.append(metrics)

        return pd.DataFrame(records)