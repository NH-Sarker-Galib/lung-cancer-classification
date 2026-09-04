import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import timm

device = torch.device("cpu")

# 🔥 UPDATED MODELS
models_to_compare = [
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "deit3_base_patch16_224",   # NEW
    "swin_tiny_patch4_window7_224",
    "swinv2_tiny_window8_256",  # NEW
    "beit_base_patch16_224",    # ADD THIS
    "levit_128s"                # ADD THIS
]

results = []

for model_name in models_to_compare:

    print(f"\nEvaluating: {model_name}")

    # ===== IMAGE SIZE FIX =====
    if "swinv2" in model_name:
        IMG_SIZE = 256
    else:
        IMG_SIZE = 224

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    val_dataset = datasets.ImageFolder(
        "CT_dataset_split/val",
        transform=transform
    )

    val_loader = DataLoader(val_dataset, batch_size=8)

    # ===== MODEL =====
    model = timm.create_model(model_name, pretrained=False)

    # ===== CLASSIFIER FIX =====
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        model.head = nn.Linear(model.head.in_features, 2)

    elif hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, 2)

    # ===== LOAD WEIGHT =====
    weight_file = f"best_{model_name}.pth"
    state_dict = torch.load(weight_file, map_location=device)

    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            # 🔥 SWINV2 FIX
            if outputs.dim() == 4:
                outputs = outputs.mean([1, 2])

            # LeViT / DeiT distillation
            if isinstance(outputs, tuple):
                outputs = (outputs[0] + outputs[1]) / 2

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
auc = roc_auc_score(all_labels, all_probs)

    results.append([model_name, acc, f1, auc])

# ===== RESULTS =====
print("\n===== FINAL COMPARISON =====")
for r in results:
    print(f"{r[0]} | Acc={r[1]:.4f} | F1={r[2]:.4f} | AUC={r[3]:.4f}")