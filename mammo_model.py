import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MammoVisionClassifier(nn.Module):
    """
    Pure Vision Mammography Classifier for predicting Benign vs. Malignant tumors.

    Architectural Specifications:
    - Pure vision architecture: model.image_encoder (ResNet-50 backbone) + classifier head.
    - Strictly decoupled from density: density is never used as input or prediction target.
      (Density is evaluated post-hoc during inference to assess model fairness across subgroups).
    - No text features or text prompts.
    - Penultimate feature activations (layer4) hooked for vision XAI methods (Grad-CAM++, IG, etc.).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.image_encoder = models.resnet50(weights=weights)
        num_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        # Pure vision binary classifier: Benign (0) vs. Malignant (1)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Penultimate activation maps (layer4)
        x = self.image_encoder.conv1(x)
        x = self.image_encoder.bn1(x)
        x = self.image_encoder.relu(x)
        x = self.image_encoder.maxpool(x)

        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)

        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        return x

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extracts 2048-dim global average pooled feature vectors."""
        feat_map = self.forward_features(img)
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        return pooled

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass taking only the mammogram image tensor."""
        pooled = self.extract_features(img)
        logits = self.classifier(pooled)
        return logits
