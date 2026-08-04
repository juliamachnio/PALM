# pycls/datasets/brisc2025.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

class BRISC2025(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        default_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.no_aug = False
        self.default_transform = default_transform

        split_dir = os.path.join(root, "train" if train else "test")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"BRISC2025 split dir not found: {split_dir}")

        self.ds = ImageFolder(root=split_dir, transform=None)

        # Expose common attributes used elsewhere
        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx
        self.samples = self.ds.samples
        self.targets = [y for _, y in self.ds.samples]
        self.imgs = self.samples

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        path, target = self.ds.samples[index]
        img = Image.open(path).convert("RGB")
        if self.no_aug and self.default_transform is not None:
            img = self.default_transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, target
