import os
import shutil
import random

# Paths
base_dir = "Dataset"
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")

output_dir = base_dir

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Get files
image_files = sorted(os.listdir(images_dir))
random.shuffle(image_files)

total = len(image_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

def copy_files(file_list, split):
    for file in file_list:
        img_src = os.path.join(images_dir, file)
        mask_src = os.path.join(masks_dir, file)

        img_dst = os.path.join(output_dir, split, "images", file)
        mask_dst = os.path.join(output_dir, split, "masks", file)

        if os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)

# Copy
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset split completed!")