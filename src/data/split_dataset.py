import os
import shutil
import random

source_dir = "CT_dataset_ready"
dest_dir = "CT_dataset_split"

classes = ["benign", "malignant"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_split = int(train_ratio * len(images))
    val_split = int((train_ratio + val_ratio) * len(images))

    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]

    for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

        for img in split_imgs:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(dest_dir, split, cls, img)
            shutil.copy(src, dst)

print("Dataset split completed.")
