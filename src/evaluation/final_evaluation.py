import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import numpy as np

# =========================
# SETTINGS
# =========================

device = torch.device("cpu")

# BEST MODEL (change if needed)
MODEL_NAME = "deit_base_patch16_224"
WEIGHT_FILE = "best_deit_base_patch16_224.pth"

DATA_PATH = "CT_dataset_split/val"

# =========================
# DATA TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# =========================
# LOAD MODEL
# =========================

model = timm.create_model(MODEL_NAME, pretrained=False)

# universal classifier fix
if hasattr(model, "head"):
    if isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, 2)
    elif hasattr(model.head, "fc"):
        model.head.fc = nn.Linear(model.head.fc.in_features, 2)

elif hasattr(model, "classifier"):
    model.classifier = nn.Linear(model.classifier.in_features, 2)

model.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# =========================
# EVALUATION
# =========================

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# =========================
# METRICS
# =========================

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print("\n===== FINAL RESULTS =====")
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Benign","Malignant"],
            yticklabels=["Benign","Malignant"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# ROC CURVE
# =========================

fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

print("\nEvaluation completed.")
print("Saved files:")
print(" - confusion_matrix.png")
print(" - roc_curve.png")