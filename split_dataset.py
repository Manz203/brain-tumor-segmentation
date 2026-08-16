import os
import shutil
import random
import json
import argparse

# ───────────────────────── CLI ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="Dataset")
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--test_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42,
                     help="Fixed seed so the split is reproducible across runs")
args = parser.parse_args()

random.seed(args.seed)

base_dir = args.base_dir
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")
output_dir = base_dir

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Get files (sorted first so shuffle is deterministic given the same seed)
image_files = sorted(os.listdir(images_dir))
random.shuffle(image_files)

total = len(image_files)
train_end = int(total * args.train_ratio)
val_end = train_end + int(total * args.val_ratio)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]


def copy_files(file_list, split):
    copied = []
    skipped = []
    for file in file_list:
        img_src = os.path.join(images_dir, file)
        mask_src = os.path.join(masks_dir, file)
        img_dst = os.path.join(output_dir, split, "images", file)
        mask_dst = os.path.join(output_dir, split, "masks", file)

        if os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
            copied.append(file)
        else:
            skipped.append(file)
    return copied, skipped


manifest = {"seed": args.seed, "ratios": {
    "train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio
}}

for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    copied, skipped = copy_files(files, split)
    manifest[split] = copied
    if skipped:
        print(f"[WARN] {len(skipped)} file(s) in '{split}' had no matching mask and were skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"[INFO] {split}: {len(copied)} pairs copied")

# Save manifest so the exact split can be audited / reproduced later
manifest_path = os.path.join(output_dir, "split_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ Dataset split completed! Manifest saved to {manifest_path}")
