"""
Run this script ONCE after extracting the folder.
It updates all absolute paths to match your current directory.

Usage:
    cd Chris_Copy
    python setup_paths.py
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "my_data")

# Old path patterns to replace
OLD_PROJECT = "/workspace/FYP_RESULTS/Final-Year-Project"
OLD_WORKSPACE = "/workspace"

# ── 1. Update YAML ──
yaml_path = os.path.join(BASE_DIR, "mydata_capped.yaml")
with open(yaml_path) as f:
    content = f.read()
content = content.replace(OLD_PROJECT + "/datasets/my_data", DATASET_DIR)
for txt_name in ["train_capped.txt", "val_capped.txt", "test_capped.txt"]:
    content = content.replace(f"{OLD_WORKSPACE}/{txt_name}", os.path.join(BASE_DIR, txt_name))
with open(yaml_path, "w") as f:
    f.write(content)
print(f"Updated: {yaml_path}")

# ── 2. Update txt files (image paths) ──
for txt_name in ["train_capped.txt", "val_capped.txt", "test_capped.txt"]:
    txt_path = os.path.join(BASE_DIR, txt_name)
    with open(txt_path) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if line:
            fname = os.path.basename(line)
            new_lines.append(os.path.join(DATASET_DIR, "images", fname))
    with open(txt_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"Updated: {txt_path} ({len(new_lines)} paths)")

# ── 3. Update notebook paths ──
for nb_name in os.listdir(BASE_DIR):
    if not nb_name.endswith(".ipynb"):
        continue
    nb_path = os.path.join(BASE_DIR, nb_name)
    with open(nb_path) as f:
        content = f.read()
    content = content.replace(OLD_PROJECT, BASE_DIR)
    content = content.replace(f"{OLD_WORKSPACE}/train_capped.txt", os.path.join(BASE_DIR, "train_capped.txt"))
    content = content.replace(f"{OLD_WORKSPACE}/val_capped.txt", os.path.join(BASE_DIR, "val_capped.txt"))
    content = content.replace(f"{OLD_WORKSPACE}/test_capped.txt", os.path.join(BASE_DIR, "test_capped.txt"))
    # Also handle yaml references like data="mydata_capped.yaml"
    content = content.replace('data=\\"mydata_capped.yaml\\"', f'data=\\"{os.path.join(BASE_DIR, "mydata_capped.yaml")}\\"')
    with open(nb_path, "w") as f:
        f.write(content)
    print(f"Updated: {nb_path}")

print("\nAll paths updated! You can now run the notebooks.")
