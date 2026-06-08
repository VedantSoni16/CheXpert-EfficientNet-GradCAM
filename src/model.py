import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained EfficientNet
        weights       = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace the final classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

        # Stage 1: Freeze everything except the new head
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes the network for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)