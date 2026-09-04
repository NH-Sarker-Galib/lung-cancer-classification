import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import timm
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import cv2

# ================= SETTINGS =================
DEVICE = torch.device("cpu")

CT_MODEL_PATH = r"F:/New folder/CT_dataset/best_deit_base_patch16_224.pth"
MRI_MODEL_PATH = r"F:/New folder/MRI_dataset/best_MRI_deit_base_patch16_224.pth"

MODEL_NAME = "deit_base_patch16_224"

CT_CLASSES = ["No Cancer", "Cancer"]
MRI_CLASSES = ["Cancer", "No Cancer"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= LOAD MODEL =================
def load_model(weight_path):
    model = timm.create_model(MODEL_NAME, pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

print("Loading models...")
ct_model = load_model(CT_MODEL_PATH)
mri_model = load_model(MRI_MODEL_PATH)
print("Models loaded successfully.")

# ================= DETECT MODALITY =================
def detect_modality(path):
    path = path.lower()
    if "ct" in path:
        return "CT"
    elif "mri" in path:
        return "MRI"
    return None

# ================= GRADCAM =================
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def generate_gradcam(model, img_tensor):
    global activations, gradients

    target_layer = model.blocks[-1].norm1
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients.mean(dim=1, keepdim=True)
    cam = (grads * activations).sum(dim=2)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    return cam

# ================= GUI =================
root = tk.Tk()
root.title("AI Lung Cancer Detection System")
root.geometry("1100x750")
root.configure(bg="#eef2f3")

title = tk.Label(root,
                 text="AI-Based Lung Cancer Detection (CT & MRI)",
                 font=("Arial", 22, "bold"),
                 bg="#eef2f3")
title.pack(pady=15)

image_frame = tk.Frame(root, bg="#eef2f3")
image_frame.pack(pady=10)

original_label = tk.Label(image_frame, bg="white")
original_label.grid(row=0, column=0, padx=20)

gradcam_label = tk.Label(image_frame, bg="white")
gradcam_label.grid(row=0, column=1, padx=20)

result_frame = tk.Frame(root, bg="#eef2f3")
result_frame.pack(pady=10)

modality_label = tk.Label(result_frame, font=("Arial",14), bg="#eef2f3")
modality_label.pack()

prediction_label = tk.Label(result_frame,
                            font=("Arial",28,"bold"),
                            bg="#eef2f3")
prediction_label.pack(pady=5)

confidence_label = tk.Label(result_frame,
                            font=("Arial",14),
                            bg="#eef2f3")
confidence_label.pack()

confidence_bar = ttk.Progressbar(result_frame,
                                 length=400,
                                 mode="determinate")
confidence_bar.pack(pady=5)

warning_label = tk.Label(root,
                         font=("Arial",12),
                         fg="orange",
                         bg="#eef2f3")
warning_label.pack(pady=5)

def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    modality = detect_modality(file_path)
    if modality is None:
        messagebox.showerror("Error", "Cannot detect CT / MRI from path")
        return

    img = Image.open(file_path).convert("L")
    img_resized = img.resize((350,350))
    img_tk = ImageTk.PhotoImage(img_resized)

    original_label.config(image=img_tk)
    original_label.image = img_tk

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    if modality == "CT":
        model = ct_model
        classes = CT_CLASSES
    else:
        model = mri_model
        classes = MRI_CLASSES

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    prediction = classes[pred]
    confidence_percent = round(confidence*100,2)

    modality_label.config(text=f"Modality: {modality}")
    prediction_label.config(text=prediction)
    confidence_label.config(text=f"Confidence: {confidence_percent}%")
    confidence_bar["value"] = confidence_percent

    if prediction == "Cancer":
        prediction_label.config(fg="red")
        cam = generate_gradcam(model, img_tensor)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        original_np = cv2.resize(np.array(img), (224,224))
        original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)

        superimposed = cv2.addWeighted(original_np,0.6,heatmap,0.4,0)
        superimposed = cv2.resize(superimposed,(350,350))

        cam_img = ImageTk.PhotoImage(Image.fromarray(superimposed))
        gradcam_label.config(image=cam_img)
        gradcam_label.image = cam_img
    else:
        prediction_label.config(fg="green")
        gradcam_label.config(image="")
        warning_label.config(text="")

button = tk.Button(root,
                   text="Select CT / MRI Image",
                   font=("Arial",14),
                   bg="#2471A3",
                   fg="white",
                   command=load_image)
button.pack(pady=15)

footer = tk.Label(root,
                  text="Research Demonstration Only | Includes Grad-CAM Explainability",
                  font=("Arial",10),
                  fg="gray",
                  bg="#eef2f3")
footer.pack(side="bottom", pady=10)

root.mainloop()