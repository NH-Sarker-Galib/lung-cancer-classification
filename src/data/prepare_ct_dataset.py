import os
from PIL import Image
from tqdm import tqdm

input_dir = "CT_dataset_clean"
output_dir = "CT_dataset_ready"

classes = ["benign", "malignant"]

for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

print("Resizing images...")

for cls in classes:
    class_path = os.path.join(input_dir, cls)
    save_path = os.path.join(output_dir, cls)

    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))

        img.save(os.path.join(save_path, img_name))

print("Done.")
