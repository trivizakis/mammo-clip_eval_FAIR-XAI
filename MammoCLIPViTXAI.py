import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class MammoCLIPViTXAI:
    """
    ViT-native and Multimodal Explainability Suite for Mammo-CLIP.
    Hooks into Multi-Head Self-Attention (MHSA) blocks and spatial patch embeddings.
    """
    def __init__(self, vit_model: nn.Module, device: str = 'cuda'):
        self.model = vit_model.to(device)
        self.device = device
        self.model.eval()

        self.attn_matrices = []
        self.attn_gradients = []
        self.hooks = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        # Target Multi-Head Attention modules in ViT blocks
        for block in self.model.blocks:
            hook_fwd = block.attn.register_forward_hook(self._forward_hook)
            self.hooks.append(hook_fwd)

    def _forward_hook(self, module, inp, outp):
        # Extract softmax attention map [B, num_heads, num_tokens, num_tokens]
        # Most ViT implementations compute: attn = (q @ k.T).softmax(dim=-1)
        if hasattr(module, 'attn_map'):
            self.attn_matrices.append(module.attn_map)
            if module.attn_map.requires_grad:
                module.attn_map.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.attn_gradients.insert(0, grad)

    def reset_inspectors(self):
        self.attn_matrices.clear()
        self.attn_gradients.clear()

    # 1. ViT Transformer Attribution (Chefer et al.)
    def transformer_attribution(self, img: torch.Tensor, target_class_idx: int = 1) -> np.ndarray:
        self.reset_inspectors()
        self.model.zero_grad()
        
        logits = self.model(img)
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        num_tokens = self.attn_matrices[0].shape[-1]
        R = torch.eye(num_tokens, num_tokens, device=self.device).unsqueeze(0)

        # Propagate relevance through Transformer layers
        for A, grad in zip(self.attn_matrices, self.attn_gradients):
            # Compute positive gradient-weighted attention
            cp = (grad * A).clamp(min=0).mean(dim=1)  # Average across heads: [B, N, N]
            I = torch.eye(num_tokens, device=self.device).unsqueeze(0)
            A_bar = cp + I
            A_bar = A_bar / A_bar.sum(dim=-1, keepdim=True)
            R = torch.bmm(A_bar, R)

        # Extract CLS-to-patch attributions (exclude CLS token at index 0)
        cls_to_patches = R[0, 0, 1:]
        
        # Reshape sequence of tokens back to 2D spatial grid
        grid_dim = int(np.sqrt(cls_to_patches.shape[0]))
        heatmap = cls_to_patches.view(grid_dim, grid_dim).detach().cpu().numpy()
        
        # Upsample to match native mammogram resolution
        heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_CUBIC)
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 2. Raw Attention Rollout
    def attention_rollout(self, img: torch.Tensor, discard_ratio: float = 0.85) -> np.ndarray:
        self.reset_inspectors()
        with torch.no_grad():
            _ = self.model(img)

        num_tokens = self.attn_matrices[0].shape[-1]
        rollout = torch.eye(num_tokens, device=self.device).unsqueeze(0)

        for A in self.attn_matrices:
            A_heads = A.mean(dim=1) # [B, N, N]
            
            # Discard noisy/low-magnitude background attention
            flat = A_heads.view(A_heads.size(0), -1)
            val, _ = torch.topk(flat, int(flat.size(-1) * discard_ratio), largest=False)
            thresh = val[:, -1].view(-1, 1, 1)
            A_heads = torch.where(A_heads < thresh, torch.zeros_like(A_heads), A_heads)
            
            # Re-normalize row probabilities
            A_heads = A_heads / (A_heads.sum(dim=-1, keepdim=True) + 1e-8)
            I = torch.eye(num_tokens, device=self.device).unsqueeze(0)
            rollout = torch.bmm(0.5 * A_heads + 0.5 * I, rollout)

        cls_rollout = rollout[0, 0, 1:].detach().cpu().numpy()
        grid_dim = int(np.sqrt(cls_rollout.shape[0]))
        heatmap = cls_rollout.reshape(grid_dim, grid_dim)
        heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_CUBIC)
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 3. Vision Patch Token Saliency (Unpooled Feature Norm)
    def patch_token_saliency(self, img: torch.Tensor) -> np.ndarray:
        """
        Calculates vision feature activation norm from unpooled patch tokens.
        """
        with torch.no_grad():
            patch_tokens = self.model.forward_features(img)
            if patch_tokens.dim() == 3:
                patch_tokens = patch_tokens[:, 1:, :]  # Drop CLS token
            sim_map = torch.norm(patch_tokens, dim=-1).squeeze(0)  # [N]

        grid_dim = int(np.sqrt(sim_map.shape[0]))
        heatmap = sim_map.view(grid_dim, grid_dim).cpu().numpy()
        heatmap = np.clip(heatmap, 0, None)
        heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_CUBIC)
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 4. Multi-Head Attention Entropy (Spatial Uncertainty & Lesion Focus)
    def attention_entropy_map(self, img: torch.Tensor) -> np.ndarray:
        self.reset_inspectors()
        with torch.no_grad():
            _ = self.model(img)

        # Examine deep layers where task abstraction occurs (final block)
        final_attn = self.attn_matrices[-1][0] # [heads, N, N]
        
        # Exclude CLS token; compute entropy over spatial patch targets
        spatial_attn = final_attn[:, 1:, 1:]
        entropy = -torch.sum(spatial_attn * torch.log(spatial_attn + 1e-9), dim=-1) # [heads, N_patches]
        mean_entropy = entropy.mean(dim=0).cpu().numpy() # [N_patches]

        grid_dim = int(np.sqrt(mean_entropy.shape[0]))
        h_map = mean_entropy.reshape(grid_dim, grid_dim)
        
        # Invert: lesions correspond to collapsed/low entropy (sharp focus)
        focus_map = h_map.max() - h_map
        focus_map = cv2.resize(focus_map, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_CUBIC)
        return (focus_map - focus_map.min()) / (focus_map.max() - focus_map.min() + 1e-8)