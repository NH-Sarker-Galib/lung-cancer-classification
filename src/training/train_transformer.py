import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from collections import Counter
import timm
import sys

# ===== DEVICE =====
device = torch.device("cpu")

# ===== MODEL NAME =====
model_name = sys.argv[1]
print(f"\nTraining model: {model_name}")

# ===== AUTO IMAGE SIZE (IMPORTANT) =====
if "swinv2" in model_name:
    IMG_SIZE = 256
else:
    IMG_SIZE = 224

print(f"Using image size: {IMG_SIZE}")

# ===== DATA =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder("CT_dataset_split/train", transform=transform)
val_dataset   = datasets.ImageFolder("CT_dataset_split/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# ===== SHOW CLASS INFO =====
print("Class mapping:", train_dataset.class_to_idx)

labels_list = [label for _, label in train_dataset.samples]
class_counts = Counter(labels_list)
print("Class counts:", class_counts)

# ===== COMPUTE CLASS WEIGHTS =====
num_class0 = class_counts[0]
num_class1 = class_counts[1]

weight_class0 = 1.0
weight_class1 = num_class0 / num_class1

class_weights = torch.tensor([weight_class0, weight_class1]).to(device)
print("Using class weights:", class_weights)

# ===== CREATE MODEL =====
model = timm.create_model(
    model_name,
    pretrained=True
)

# force classifier (VERY IMPORTANT FOR SWINV2)
if hasattr(model, "head") and hasattr(model.head, "in_features"):
    model.head = nn.Linear(model.head.in_features, 2)

# ===== CLASSIFIER FIX (ROBUST) =====

# LeViT
if "levit" in model_name:
    model.head = nn.Linear(model.head.linear.in_features, 2)
    model.head_dist = nn.Linear(model.head_dist.linear.in_features, 2)

# Swin / SwinV2
elif "swin" in model_name:
    if hasattr(model.head, "in_features"):
        model.head = nn.Linear(model.head.in_features, 2)

# ViT / DeiT / DeiT v3 / BEiT
elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
    model.head = nn.Linear(model.head.in_features, 2)

# Fallback
elif hasattr(model, "classifier"):
    model.classifier = nn.Linear(model.classifier.in_features, 2)

model = model.to(device)

# ===== DEBUG (ADD HERE) =====
print("Model output test shape:")
test = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
out = model(test)
print(out.shape)

# ===== TRAINING =====
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0
epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

       outputs = model(images)

# FIX FOR SWINV2
if outputs.dim() == 4:
    outputs = outputs.mean([1, 2])

        # Handle distillation outputs (DeiT / LeViT)
        if isinstance(outputs, tuple):
            outputs = (outputs[0] + outputs[1]) / 2

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

           outputs = model(images)

if outputs.dim() == 4:
    outputs = outputs.mean([1, 2])

            if isinstance(outputs, tuple):
                outputs = (outputs[0] + outputs[1]) / 2

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {total_loss/len(train_loader):.4f}")
    print(f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"best_{model_name}.pth")
        print("Best model saved.")

print("\nTraining finished.")