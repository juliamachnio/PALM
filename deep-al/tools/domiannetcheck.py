from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

# Config
DOMAIN = "real"
TARGET_DIR = f"../datasets/domainnet/{DOMAIN}"

# Load all data (only 'default' config is available, contains all domains)
dataset = load_dataset("wltjr1007/DomainNet")  # no name="real"

# Create split folders
for split in ["train", "test"]:
    split_path = os.path.join(TARGET_DIR, split)
    os.makedirs(split_path, exist_ok=True)

    for item in tqdm(dataset[split]):
        if item["domain"] != DOMAIN:
            continue
        label = item["class"].replace("/", "_")  # sanitize
        img = item["image"]
        img_id = item["id"]

        class_dir = os.path.join(split_path, label)
        os.makedirs(class_dir, exist_ok=True)

        img.save(os.path.join(class_dir, f"{img_id}.png"))
