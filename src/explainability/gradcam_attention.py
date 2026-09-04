import torch
import torch.nn.functional as F
import cv2
import numpy as np

# =========================
# GRADCAM FOR TRANSFORMER
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):

        output = self.model(input_tensor)

        if isinstance(output, tuple):
            output = (output[0] + output[1]) / 2

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.detach().cpu().numpy()
        return cam


# =========================
# ATTENTION MAP (ViT/DeiT)
# =========================
def get_attention_map(model, input_tensor):

    with torch.no_grad():
        _ = model(input_tensor)

    # last block attention
    attn = model.blocks[-1].attn.attn

    attn = attn.mean(dim=1).squeeze()  # avg heads
    attn = attn[0, 1:]  # remove CLS token

    size = int(np.sqrt(attn.shape[0]))
    attn = attn.reshape(size, size)

    attn = attn.cpu().numpy()
    attn = cv2.resize(attn, (224, 224))

    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-8)

    return attn


# =========================
# COMBINE HEATMAPS
# =========================
def combine_maps(gradcam_map, attn_map, alpha=0.5):
    combined = alpha * gradcam_map + (1 - alpha) * attn_map
    combined = combined - combined.min()
    combined = combined / (combined.max() + 1e-8)
    return combined


# =========================
# OVERLAY FUNCTION
# =========================
def overlay_heatmap(img, heatmap):

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + np.float32(img)
    overlay = overlay / np.max(overlay)

    return np.uint8(255 * overlay)