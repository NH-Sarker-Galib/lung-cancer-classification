import h5py
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

file_path = "all_patches.hdf5"
output_dir = "CT_dataset_clean"

# Create folders
os.makedirs(os.path.join(output_dir, "benign"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "malignant"), exist_ok=True)

with h5py.File(file_path, "r") as f:
    images = f["ct_slices"][:]
    labels = f["slice_class"][:].flatten()

print("Exporting images...")

for i in tqdm(range(len(images))):
    img = images[i]
    label = labels[i]

    # Normalize image to 0-255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    img_pil = Image.fromarray(img)

    if label == 0:
        save_path = os.path.join(output_dir, "benign", f"img_{i}.png")
    else:
        save_path = os.path.join(output_dir, "malignant", f"img_{i}.png")

    img_pil.save(save_path)

print("Done exporting.")
