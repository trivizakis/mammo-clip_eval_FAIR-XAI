import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.segmentation import slic


class MammographyXAI:
    """
    Comprehensive Vision-Only Explainable AI (XAI) Suite for Mammography.
    Implements gradient-based, path-attribution, and perturbation-based vision XAI methods:
    - Grad-CAM
    - Grad-CAM++ (Second/Third order gradient localization)
    - Integrated Gradients (Vision-only pixel-level path attributions)
    - Occlusion Sensitivity (Spatial perturbation mapping)
    - RISE (Randomized Input Sampling for Explanation)
    - Superpixel SHAP (SLIC-based Shapley value attribution)
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def grad_cam(self, img: torch.Tensor) -> np.ndarray:
        """Grad-CAM for binary lesion classification."""
        self.model.zero_grad()
        img = img.clone().detach().requires_grad_(True)
        out = self.model(img)
        out.backward()

        grads = self.model.gradients.detach()  # [1, C, H, W]
        acts = self.model.activations.detach()  # [1, C, H, W]

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze().clamp(min=0)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

    def grad_cam_plus_plus(self, img: torch.Tensor) -> np.ndarray:
        """Grad-CAM++: High-order gradient weighting for fine-grained lesion localization."""
        self.model.zero_grad()
        img = img.clone().detach().requires_grad_(True)
        out = self.model(img)
        out.backward()

        grads = self.model.gradients.detach()
        acts = self.model.activations.detach()

        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze().clamp(min=0)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

    def integrated_gradients(self, img: torch.Tensor, steps: int = 40) -> np.ndarray:
        """Vision Integrated Gradients: Straight path integration from black baseline."""
        baseline = torch.zeros_like(img)
        scaled_inputs = [baseline + (float(i) / steps) * (img - baseline) for i in range(steps + 1)]
        grads = []

        for step_input in scaled_inputs:
            step_input = step_input.clone().detach().requires_grad_(True)
            out = self.model(step_input)
            self.model.zero_grad()
            out.backward()
            grads.append(step_input.grad.detach())

        avg_grads = torch.mean(torch.stack(grads), dim=0)
        integrated_grad = (img - baseline) * avg_grads
        attributions = integrated_grad.detach().abs().sum(dim=(0, 1)).cpu().numpy()
        attributions = cv2.GaussianBlur(attributions, (9, 9), 0)
        return (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)

    def occlusion_sensitivity(self, img: torch.Tensor, patch_size: int = 64, stride: int = 32) -> np.ndarray:
        """Systematic sliding-window occlusion sensitivity analysis."""
        _, _, H, W = img.shape
        with torch.no_grad():
            baseline_prob = torch.sigmoid(self.model(img)).item()

        heatmap = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H - patch_size, stride):
            for x in range(0, W - patch_size, stride):
                perturbed = img.clone()
                perturbed[:, :, y:y+patch_size, x:x+patch_size] = 0.0
                with torch.no_grad():
                    perturbed_prob = torch.sigmoid(self.model(perturbed)).item()
                drop = baseline_prob - perturbed_prob
                heatmap[y:y+patch_size, x:x+patch_size] += drop
                counts[y:y+patch_size, x:x+patch_size] += 1.0

        valid = counts > 0
        heatmap[valid] /= counts[valid]
        heatmap = np.clip(heatmap, 0, None)
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    def rise(self, img: torch.Tensor, num_masks: int = 300, p: float = 0.1, cell_size: int = 32) -> np.ndarray:
        """Randomized Input Sampling for Explanation (RISE)."""
        _, _, H, W = img.shape
        h_grid, w_grid = int(np.ceil(H / cell_size)), int(np.ceil(W / cell_size))

        raw_masks = (torch.rand(num_masks, 1, h_grid, w_grid, device=self.device) < p).float()
        upsampled_masks = F.interpolate(raw_masks, size=(H, W), mode='bilinear', align_corners=False)

        heatmap = torch.zeros(H, W, device=self.device)
        batch_size = 16
        with torch.no_grad():
            for i in range(0, num_masks, batch_size):
                m_batch = upsampled_masks[i:i+batch_size]
                masked_imgs = img * m_batch
                preds = torch.sigmoid(self.model(masked_imgs)).squeeze(-1)

                weights = preds.view(-1, 1, 1, 1)
                heatmap += (m_batch * weights).sum(dim=0).squeeze()

        sal = heatmap.cpu().numpy()
        return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    def superpixel_shap(self, img: torch.Tensor, n_segments: int = 80) -> np.ndarray:
        """Superpixel SHAP attribution via SLIC oversegmentation."""
        np_img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        segments = slic(np_img, n_segments=n_segments, compactness=15, sigma=1)
        unique_segs = np.unique(segments)
        num_segs = len(unique_segs)

        with torch.no_grad():
            base_score = torch.sigmoid(self.model(img)).item()

        shap_values = np.zeros(num_segs)
        samples = 100
        for _ in range(samples):
            active = np.random.choice([0, 1], size=num_segs, p=[0.5, 0.5])
            mask = np.isin(segments, unique_segs[active == 1])
            masked_img = img.clone()
            masked_img[:, :, ~mask] = 0.0

            with torch.no_grad():
                prob = torch.sigmoid(self.model(masked_img)).item()
            contribution = prob - base_score
            for idx, act in enumerate(active):
                shap_values[idx] += contribution if act else -contribution

        shap_values /= samples
        attribution_map = np.zeros_like(segments, dtype=np.float32)
        for idx, seg_id in enumerate(unique_segs):
            attribution_map[segments == seg_id] = shap_values[idx]

        attribution_map = np.clip(attribution_map, 0, None)
        return (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)