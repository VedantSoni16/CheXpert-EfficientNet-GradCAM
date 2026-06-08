import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Binary classifier has one logit
        self.model.zero_grad()
        score = output[0][0]   # single logit
        score.backward()

        # Pool gradients across spatial dimensions → importance weights per channel
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)

        # ReLU — only keep features that positively contribute to the positive class
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def get_prediction(self, input_tensor):
        """Returns probability and binary label for display in Streamlit."""
        self.model.eval()
        with torch.no_grad():
            logit = self.model(input_tensor)[0][0]
            prob  = torch.sigmoid(logit).item()
        return prob, int(prob >= 0.5)


def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap onto the original image from a file path.
    """
    orig_img        = cv2.imread(img_path)
    orig_img        = cv2.resize(orig_img, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    color_heatmap   = cv2.applyColorMap(heatmap_uint8, colormap)
    overlayed       = cv2.addWeighted(color_heatmap, alpha, orig_img, 1 - alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)


def overlay_heatmap_from_array(img_array, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap using a PIL/numpy image array in memory.
    """
    orig_img        = cv2.resize(np.array(img_array), (224, 224))
    orig_bgr        = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    color_heatmap   = cv2.applyColorMap(heatmap_uint8, colormap)
    overlayed       = cv2.addWeighted(color_heatmap, alpha, orig_bgr, 1 - alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    from model import ChestXrayModel
    model  = ChestXrayModel(freeze_backbone=False)
    target = model.backbone.features[-1]
    cam    = GradCAM(model, target)
    print("GradCAM hook registration successful.")
    print(f"Target layer: {target.__class__.__name__}")