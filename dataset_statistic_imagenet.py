import os
from collections import Counter

# Paths
IMAGENET_DIR = "/home/ju-ma/PycharmProjects/rankk/scan/imagenet/ILSVRC/Data/CLS-LOC"
SUBSET_DIR = "/home/ju-ma/PycharmProjects/rankk/scan/data/imagenet_subsets"


def load_classes(subset_file):
    with open(subset_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def count_images(split_dir, classes):
    counts = {}
    for cls in classes:
        class_dir = os.path.join(split_dir, cls)
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: Class {cls} not found in {split_dir}")
            counts[cls] = 0
            continue
        counts[cls] = len([f for f in os.listdir(class_dir) if f.lower().endswith(".jpeg")])
    return counts


def stats_imagenet_subset(subset_name):
    subset_file = os.path.join(SUBSET_DIR, f"{subset_name}.txt")
    classes = load_classes(subset_file)

    train_counts = count_images(os.path.join(IMAGENET_DIR, "train"), classes)
    val_counts = count_images(os.path.join(IMAGENET_DIR, "val"), classes)

    print(f"\n=== ImageNet {subset_name.upper()} statistics ===")
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"Total train images: {total_train}")
    print(f"Total val images:   {total_val}")

    for cls in classes:
        print(f"{cls}: train={train_counts[cls]}, val={val_counts[cls]}")
    print()


if __name__ == "__main__":
    for subset in ["imagenet_50", "imagenet_100", "imagenet_200"]:
        stats_imagenet_subset(subset)
