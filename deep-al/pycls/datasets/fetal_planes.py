# pycls/datasets/fetal_planes.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CLASSES = ["Fetal_abdomen", "Fetal_brain", "Fetal_femur", "Fetal_thorax", "Maternal_cervix", "Other"]


class FetalPlanes(Dataset):
    """
    Fetal Planes DB classification dataset (6 classes).

    Expected structure:
      root/
        Train/<class>/*.png
        Test/<class>/*.png

    Classes: Fetal_abdomen, Fetal_brain, Fetal_femur, Fetal_thorax, Maternal_cervix, Other
    """

    CLASSES = CLASSES
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(self, root, train=True, transform=None, default_transform=None, **kwargs):
        super().__init__()
        self.root = root
        self.transform = transform
        self.default_transform = default_transform
        self.no_aug = False

        split_dir = os.path.join(root, "Train" if train else "Test")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"[FetalPlanes] Missing split dir: {split_dir}")

        self.ds = ImageFolder(root=split_dir, transform=None)
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
