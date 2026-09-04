import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ======================
# DEVICE
# ======================
device = torch.device("cpu")
print("Using device:", device)

# ======================
# DATA PATHS (YOUR FOLDERS)
# ======================
train_dir = "CT_dataset_split/train"
val_dir   = "CT_dataset_split/val"

# ======================
# TRANSFORMS
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# DATASET
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# ======================
# MODEL (ViT)
# ======================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=2
)

model = model.to(device)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ======================
# TRAINING SETTINGS
# ======================
epochs = 5
best_acc = 0

# ======================
# TRAIN + EVALUATE LOOP
# ======================
for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    # ---- TRAIN ----
    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print("Loss:", round(avg_loss, 4))

    # ---- VALIDATION ----
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val F1: {f1:.4f}")
    print(f"Val AUC: {auc:.4f}")

    # ---- SAVE BEST MODEL ----
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_vit_model.pth")
        print("Best model saved.")

print("\nTraining finished.")