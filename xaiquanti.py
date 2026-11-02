import numpy as np
import torch
import torch.nn as nn
try:
    from scipy.integrate import trapezoid as trapz_fn
except ImportError:
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


class XAIQuantitativeAuditor:
    """
    Quantitative explainability auditor measuring attribution sparsity,
    spatial energy concentration, and perturbation-based model faithfulness.
    """
    @staticmethod
    def gini_sparsity(saliency: np.ndarray) -> float:
        flat = np.sort(saliency.flatten())
        n = flat.shape[0]
        index = np.arange(1, n + 1)
        return float(((np.sum((2 * index - n - 1) * flat)) / (n * np.sum(flat) + 1e-12)))

    @staticmethod
    def energy_concentration(saliency: np.ndarray, top_k_ratio: float = 0.05) -> float:
        flat = saliency.flatten()
        k = int(len(flat) * top_k_ratio)
        top_k_sum = np.sum(np.partition(flat, -k)[-k:] ** 2)
        total_sum = np.sum(flat ** 2) + 1e-12
        return float(top_k_sum / total_sum)

    @staticmethod
    def faithfulness_audc(model: nn.Module, img: torch.Tensor, saliency: np.ndarray, steps: int = 10) -> float:
        """
        Measures decay of tumor prediction confidence as most salient pixels are progressively removed.
        Returns Area Under Deletion Curve (AUDC).
        """
        flat_indices = np.argsort(saliency.flatten())[::-1]
        probs = []
        perturbed = img.clone()
        _, _, H, W = img.shape

        chunk = len(flat_indices) // steps
        with torch.no_grad():
            for i in range(steps):
                probs.append(torch.sigmoid(model(perturbed)).item())
                idx_to_mask = flat_indices[i * chunk : (i + 1) * chunk]
                y_coords, x_coords = np.unravel_index(idx_to_mask, (H, W))
                perturbed[:, :, y_coords, x_coords] = 0.0

        return float(trapz_fn(probs, dx=1.0 / steps))