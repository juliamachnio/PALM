import os
import pickle
from collections import Counter

DATASET_DIR = "{your_path}/scan/datasets"

def load_cifar_batch(file_path, encoding="latin1"):
    """Load a single CIFAR batch."""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding=encoding)
    # Some CIFAR files use 'labels', CIFAR-100 uses 'fine_labels'
    labels = batch.get("labels") or batch.get("fine_labels")
    return labels

def stats_cifar10():
    cifar10_path = os.path.join(DATASET_DIR, "cifar-10", "cifar-10-batches-py")
    label_names = None
    with open(os.path.join(cifar10_path, "batches.meta"), "rb") as f:
        meta = pickle.load(f, encoding="latin1")
        label_names = meta["label_names"]

    all_labels = []
    # Training batches
    for i in range(1, 6):
        all_labels.extend(load_cifar_batch(os.path.join(cifar10_path, f"data_batch_{i}")))
    # Test batch
    all_labels.extend(load_cifar_batch(os.path.join(cifar10_path, "test_batch")))

    counts = Counter(all_labels)
    print("CIFAR-10 class distribution:")
    for i, name in enumerate(label_names):
        print(f"{name}: {counts[i]}")
    print(f"Total samples: {sum(counts.values())}\n")

def stats_cifar100():
    cifar100_path = os.path.join(DATASET_DIR, "cifar-100", "cifar-100-python")
    with open(os.path.join(cifar100_path, "meta"), "rb") as f:
        meta = pickle.load(f, encoding="latin1")
        fine_label_names = meta["fine_label_names"]

    all_labels = []
    for split in ["train", "test"]:
        all_labels.extend(load_cifar_batch(os.path.join(cifar100_path, split)))

    counts = Counter(all_labels)
    print("CIFAR-100 class distribution:")
    for i, name in enumerate(fine_label_names):
        print(f"{name}: {counts[i]}")
    print(f"Total samples: {sum(counts.values())}\n")

if __name__ == "__main__":
    stats_cifar10()
    stats_cifar100()
