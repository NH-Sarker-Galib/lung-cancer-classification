import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# =========================
# SETTINGS
# =========================
MODEL_NAME = "deit_base_patch16_224"
WEIGHT_PATH = "best_deit_base_patch16_224.pth"
import glob

# auto pick first validation image
IMAGE_PATH = glob.glob("CT_dataset_split/val/*/*.png")[0]

print("Using image:", IMAGE_PATH)

device = torch.device("cpu")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# =========================
# LOAD MODEL
# =========================
model = timm.create_model(MODEL_NAME, pretrained=False)

# classifier replace
if hasattr(model, "head") and isinstance(model.head, nn.Linear):
    model.head = nn.Linear(model.head.in_features, 2)

model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()
model.to(device)

print("Model loaded.")

# =========================
# HOOKS FOR GRADCAM
# =========================
activations = None
gradients = None

def forward_hook(module, inp, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# transformer last block
target_layer = model.blocks[-1].norm1

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# =========================
# LOAD IMAGE
# =========================
img_pil = Image.open(IMAGE_PATH).convert("L")
input_tensor = transform(img_pil).unsqueeze(0).to(device)

# =========================
# FORWARD + BACKWARD
# =========================
output = model(input_tensor)
pred_class = torch.argmax(output)

model.zero_grad()
output[0, pred_class].backward()

# =========================
# PROCESS GRADCAM
# =========================
grads = gradients.mean(dim=1, keepdim=True)
cam = (activations * grads).sum(dim=-1)

cam = cam.squeeze().detach().numpy()
cam = np.maximum(cam,0)
cam = cv2.resize(cam, (224,224))

cam = (cam - cam.min()) / (cam.max() - cam.min())
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

# original image
img_np = np.array(img_pil.resize((224,224)))
img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

overlay = cv2.addWeighted(img_np,0.5,heatmap,0.5,0)

# =========================
# SAVE OUTPUT
# =========================
cv2.imwrite("gradcam_result.png", overlay)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("GradCAM Result")
plt.axis("off")
plt.show()

print("GradCAM saved as gradcam_result.png")