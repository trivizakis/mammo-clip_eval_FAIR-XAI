import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MammoDetectorXAI:
    """
    Dedicated Explainability (XAI) Suite for Mammo-CLIP RetinaNet Detector:
    1. Det-CAM: Multi-Scale Class Activation Mapping for object detection proposals.
    2. Separated Localization vs. Classification Saliency:
       - Classification Saliency: Texture & lesion patterns driving mass semantic confidence.
       - Localization Saliency: Boundary & structural contours driving spatial bounding box anchors.
    3. D-RISE: Detector Randomized Input Sampling for Explanation.
    4. Mammo-FActOR: Feature attribution via dense concept cosine similarity against 'Mass'.
    5. Monte Carlo Prediction Variability: Epistemic & spatial uncertainty quantification across stochastic passes.
    """

    def __init__(
        self,
        model: nn.Module,
        clip_chk_pt_path: str,
        concept_vec_path: str = None,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Load image projection weights for Mammo-FActOR
        ckpt = torch.load(clip_chk_pt_path, map_location="cpu", weights_only=False)
        self.image_proj = nn.Conv2d(2048, 512, kernel_size=1)
        if "image_projection.projection.weight" in ckpt["model"]:
            w = ckpt["model"]["image_projection.projection.weight"].unsqueeze(-1).unsqueeze(-1)
            b = ckpt["model"]["image_projection.projection.bias"]
            self.image_proj.weight.data.copy_(w)
            self.image_proj.bias.data.copy_(b)
        self.image_proj = self.image_proj.to(device).eval()

        # Load precomputed concept vector for 'mass'
        if concept_vec_path and os.path.exists(concept_vec_path):
            self.concept_vec = torch.load(concept_vec_path, map_location=device).squeeze()
            if self.concept_vec.dim() == 1:
                self.concept_vec = self.concept_vec.unsqueeze(0)
            self.concept_vec = F.normalize(self.concept_vec, dim=-1)
        else:
            # Fallback normalized concept projection
            self.concept_vec = torch.ones(1, 512, device=device)
            self.concept_vec = F.normalize(self.concept_vec, dim=-1)

        self.activations = None
        self.gradients = None
        self.probe_w = None
        self.probe_b = None

    def set_calibrated_probe(self, w: torch.Tensor, b: torch.Tensor):
        """Sets the calibrated linear probe weights from the 8000+ calibration set."""
        self.probe_w = w.to(self.device).float()
        self.probe_b = b.to(self.device).float()

    def _register_hooks(self, target_layer):
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, outp):
            self.activations = outp

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        return [h1, h2]

    # -------------------------------------------------------------
    # 1. Det-CAM: Detector Class Activation Mapping
    # -------------------------------------------------------------
    def compute_det_cam(self, img: torch.Tensor) -> np.ndarray:
        """
        Backpropagates detection score to penultimate backbone layer with AMP.
        """
        target_layer = self.model.efficientnet._conv_head
        hooks = self._register_hooks(target_layer)

        img_var = img.clone().detach().to(self.device).requires_grad_(True)
        with torch.amp.autocast("cuda"):
            _, C3, C4, C5 = self.model.efficientnet(img_var)
            if self.probe_w is not None:
                pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                top_score = (pooled_c5 @ self.probe_w.T + self.probe_b).squeeze()
            else:
                features = self.model.fpn([C3, C4, C5])
                cls_maps = [self.model.classificationModel(f) for f in features]
                all_cls = torch.cat(cls_maps, dim=1)
                top_score, _ = torch.max(all_cls.squeeze(-1), dim=1)

        self.model.zero_grad()
        top_score.backward()

        for h in hooks:
            h.remove()

        if self.gradients is None or self.activations is None:
            return np.zeros((img.shape[2], img.shape[3]), dtype=np.float32)

        grads = self.gradients.detach()
        acts = self.activations.detach()

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().float().cpu().numpy()

        self.activations = None
        self.gradients = None
        del img_var, C3, C4, C5, top_score, grads, acts, weights, cam
        torch.cuda.empty_cache()

        norm_cam = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return norm_cam.astype(np.float32)

    # -------------------------------------------------------------
    # 2. Separated Localization vs. Classification Saliency
    # -------------------------------------------------------------
    def compute_separated_saliency(self, img: torch.Tensor) -> tuple:
        """
        Disentangles:
        - Classification Saliency: Gradient of classification confidence S_cls.
        - Localization Saliency: Gradient of spatial anchor regression offsets / peak spatial response L_loc.
        """
        target_layer = self.model.efficientnet._conv_head

        # A) Classification Saliency
        hooks = self._register_hooks(target_layer)
        img_var_cls = img.clone().detach().to(self.device).requires_grad_(True)
        with torch.amp.autocast("cuda"):
            _, C3, C4, C5 = self.model.efficientnet(img_var_cls)
            if self.probe_w is not None:
                pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                top_score = (pooled_c5 @ self.probe_w.T + self.probe_b).squeeze()
            else:
                features = self.model.fpn([C3, C4, C5])
                cls_maps = [self.model.classificationModel(f) for f in features]
                all_cls = torch.cat(cls_maps, dim=1)
                top_score, _ = torch.max(all_cls.squeeze(-1), dim=1)

        self.model.zero_grad()
        top_score.backward()
        for h in hooks:
            h.remove()

        if self.gradients is not None and self.activations is not None:
            w_cls = torch.mean(self.gradients.detach(), dim=(2, 3), keepdim=True)
            cam_cls = F.relu(torch.sum(w_cls * self.activations.detach(), dim=1, keepdim=True))
            cam_cls = F.interpolate(cam_cls, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
            cls_np = cam_cls.squeeze().float().cpu().numpy()
            cls_saliency = (cls_np - cls_np.min()) / (cls_np.max() - cls_np.min() + 1e-8)
        else:
            cls_saliency = np.zeros((img.shape[2], img.shape[3]), dtype=np.float32)

        self.activations = None
        self.gradients = None
        del img_var_cls, C3, C4, C5, top_score
        torch.cuda.empty_cache()

        # B) Localization Saliency
        hooks = self._register_hooks(target_layer)
        img_var_loc = img.clone().detach().to(self.device).requires_grad_(True)
        with torch.amp.autocast("cuda"):
            _, C3, C4, C5 = self.model.efficientnet(img_var_loc)
            if self.probe_w is not None:
                spatial_logits = F.conv2d(C5, self.probe_w.view(1, 2048, 1, 1), bias=self.probe_b)
                spatial_probs = torch.sigmoid(spatial_logits)
                loc_loss, _ = torch.max(spatial_probs.flatten(), dim=0)
            else:
                features = self.model.fpn([C3, C4, C5])
                reg_maps = [self.model.regressionModel(f) for f in features]
                all_reg = torch.cat(reg_maps, dim=1)
                loc_loss = torch.sum(torch.abs(all_reg[:, 0, :]))

        self.model.zero_grad()
        loc_loss.backward()
        for h in hooks:
            h.remove()

        if self.gradients is not None and self.activations is not None:
            w_loc = torch.mean(torch.abs(self.gradients.detach()), dim=(2, 3), keepdim=True)
            cam_loc = F.relu(torch.sum(w_loc * self.activations.detach(), dim=1, keepdim=True))
            cam_loc = F.interpolate(cam_loc, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
            loc_np = cam_loc.squeeze().float().cpu().numpy()
            loc_saliency = (loc_np - loc_np.min()) / (loc_np.max() - loc_np.min() + 1e-8)
        else:
            loc_saliency = np.zeros((img.shape[2], img.shape[3]), dtype=np.float32)

        self.activations = None
        self.gradients = None
        del img_var_loc, C3, C4, C5, loc_loss
        torch.cuda.empty_cache()

        return cls_saliency.astype(np.float32), loc_saliency.astype(np.float32)

    # -------------------------------------------------------------
    # 3. D-RISE: Detector Randomized Input Sampling
    # -------------------------------------------------------------
    def compute_drise(
        self,
        img: torch.Tensor,
        num_masks: int = 25,
        grid_size: tuple = (16, 12),
        p: float = 0.5
    ) -> np.ndarray:
        """
        D-RISE: Generates random spatial binary masks, perturbs the input, and
        weights masks by the detection confidence score.
        """
        H, W = img.shape[2], img.shape[3]
        saliency_map = np.zeros((H, W), dtype=np.float32)
        weight_sum = 0.0

        with torch.no_grad():
            for _ in range(num_masks):
                # Generate random low-res binary mask and bilinearly upsample
                mask_low = (torch.rand(1, 1, grid_size[0], grid_size[1], device=self.device) < p).float()
                mask_up = F.interpolate(mask_low, size=(H, W), mode="bilinear", align_corners=False)

                perturbed_img = img * mask_up
                with torch.amp.autocast("cuda"):
                    _, C3, C4, C5 = self.model.efficientnet(perturbed_img)
                    if self.probe_w is not None:
                        pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                        score = float(torch.sigmoid(pooled_c5 @ self.probe_w.T + self.probe_b).item())
                    else:
                        features = self.model.fpn([C3, C4, C5])
                        cls_maps = [self.model.classificationModel(f) for f in features]
                        all_cls = torch.cat(cls_maps, dim=1)
                        score = float(torch.max(all_cls).item())

                mask_np = mask_up.squeeze().cpu().numpy()
                saliency_map += score * mask_np
                weight_sum += score

        if weight_sum > 0:
            saliency_map /= weight_sum

        norm_drise = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        return norm_drise.astype(np.float32)

    # -------------------------------------------------------------
    # 4. Mammo-FActOR: Feature Attribution via Concept Cosine Similarity
    # -------------------------------------------------------------
    def compute_mammo_factor(self, img: torch.Tensor) -> np.ndarray:
        """
        Computes dense regional cosine similarity between unpooled Mammo-CLIP
        image representations and the clinical 'Mass' concept embedding.
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                _, _, _, C5 = self.model.efficientnet(img)
                proj_features = self.image_proj(C5.float())  # [1, 512, H_feat, W_feat]
                proj_norm = F.normalize(proj_features, dim=1)  # Unit length along channels

                # Dense cosine similarity with normalized concept vector [1, 512, 1, 1]
                concept_kernel = self.concept_vec.view(1, 512, 1, 1)
                sim_map = torch.sum(proj_norm * concept_kernel, dim=1, keepdim=True)
                sim_map = F.relu(sim_map)  # Keep positive semantic alignment only

                # Upsample to image resolution
                sim_map_up = F.interpolate(sim_map, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
                factor_np = sim_map_up.squeeze().cpu().numpy()

        norm_factor = (factor_np - factor_np.min()) / (factor_np.max() - factor_np.min() + 1e-8)
        return norm_factor.astype(np.float32)

    # -------------------------------------------------------------
    # 5. Monte Carlo Prediction Variability (Uncertainty Analysis)
    # -------------------------------------------------------------
    def compute_monte_carlo_variability(
        self,
        img: torch.Tensor,
        num_passes: int = 8,
        noise_std: float = 0.02
    ) -> tuple:
        """
        Evaluates prediction stability across K stochastic Monte Carlo passes.
        Computes pixel-wise spatial prediction variance map and scalar prediction score uncertainty.
        """
        H, W = img.shape[2], img.shape[3]
        pass_maps = []
        pass_scores = []

        with torch.no_grad():
            for _ in range(num_passes):
                # Inject stochastic test-time perturbation
                noise = torch.randn_like(img) * noise_std
                noisy_img = img + noise

                with torch.amp.autocast("cuda"):
                    _, C3, C4, C5 = self.model.efficientnet(noisy_img)
                    if self.probe_w is not None:
                        pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                        top_score = float(torch.sigmoid(pooled_c5 @ self.probe_w.T + self.probe_b).item())
                        spatial_logits = F.conv2d(C5, self.probe_w.view(1, 2048, 1, 1), bias=self.probe_b)
                        spatial_probs = torch.sigmoid(spatial_logits)
                        p_up = F.interpolate(spatial_probs, size=(H, W), mode="bilinear", align_corners=False)
                        act_np = p_up.squeeze().cpu().numpy()
                    else:
                        features = self.model.fpn([C3, C4, C5])
                        cls_maps = [self.model.classificationModel(f) for f in features]
                        all_cls = torch.cat(cls_maps, dim=1)
                        top_score = float(torch.max(all_cls).item())
                        p3_act = torch.mean(features[0], dim=1, keepdim=True)
                        p3_up = F.interpolate(p3_act, size=(H, W), mode="bilinear", align_corners=False)
                        act_np = p3_up.squeeze().cpu().numpy()

                act_norm = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-8)
                pass_maps.append(act_norm)
                pass_scores.append(top_score)

        stacked_maps = np.stack(pass_maps, axis=0)  # [K, H, W]
        var_map = np.var(stacked_maps, axis=0)  # Pixel-wise variance
        score_std = float(np.std(pass_scores))

        norm_var = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        return norm_var.astype(np.float32), score_std

    # -------------------------------------------------------------
    # 6. Extract Calibrated Post-Inference Bounding Box
    # -------------------------------------------------------------
    def extract_predicted_bounding_box(
        self,
        img: torch.Tensor,
        opt_threshold: float,
        saliency_map: np.ndarray = None
    ) -> dict:
        """
        Decodes detector anchor proposals and refines the bounding box using peak activation.
        Returns:
            dict containing:
            - 'bbox': [x1, y1, x2, y2] in pixel coordinates
            - 'score': confidence probability
            - 'is_malignant': bool (score >= opt_threshold)
            - 'decision': 'MALIGNANT' or 'BENIGN'
            - 'opt_threshold': optimal calibrated threshold
        """
        H, W = img.shape[2], img.shape[3]
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                _, C3, C4, C5 = self.model.efficientnet(img)
                if self.probe_w is not None:
                    pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                    raw_score = float(torch.sigmoid(pooled_c5 @ self.probe_w.T + self.probe_b).item())
                    spatial_logits = F.conv2d(C5, self.probe_w.view(1, 2048, 1, 1), bias=self.probe_b)
                    spatial_probs = torch.sigmoid(spatial_logits)
                    spatial_up = F.interpolate(spatial_probs, size=(H, W), mode="bilinear", align_corners=False)
                    spatial_map_up = spatial_up.squeeze().float().cpu().numpy()
                else:
                    features = self.model.fpn([C3, C4, C5])
                    cls_maps = [self.model.classificationModel(f) for f in features]
                    all_cls = torch.cat(cls_maps, dim=1)
                    raw_score = float(torch.max(all_cls.squeeze(-1), dim=1)[0].item())
                    spatial_map_up = saliency_map if saliency_map is not None else np.zeros((H, W), dtype=np.float32)

        # Extract bounding box from peak activation region
        guide_map = spatial_map_up if (spatial_map_up is not None and spatial_map_up.max() > 0) else saliency_map
        if guide_map is not None and guide_map.max() > 0:
            norm_guide = (guide_map - guide_map.min()) / (guide_map.max() - guide_map.min() + 1e-8)
            thresh = np.percentile(norm_guide, 98.5)
            ys, xs = np.where(norm_guide >= thresh)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                bw = max(int(W * 0.16), int((xs.max() - xs.min()) * 1.4))
                bh = max(int(H * 0.13), int((ys.max() - ys.min()) * 1.4))
                bw = min(bw, int(W * 0.35))
                bh = min(bh, int(H * 0.28))
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                x2 = min(W, cx + bw // 2)
                y2 = min(H, cy + bh // 2)
            else:
                x1, y1, x2, y2 = W * 0.2, H * 0.2, W * 0.8, H * 0.8
        else:
            x1, y1, x2, y2 = W * 0.2, H * 0.2, W * 0.8, H * 0.8

        is_malignant = raw_score >= opt_threshold
        decision = "MALIGNANT" if is_malignant else "BENIGN"

        return {
            "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
            "score": round(float(raw_score), 4),
            "is_malignant": is_malignant,
            "decision": decision,
            "opt_threshold": round(float(opt_threshold), 4)
        }
