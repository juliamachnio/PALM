import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pycls.datasets.utils import load_features

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class ISIC2019(Dataset):
    """
    Folder-based ISIC2019 loader.

    Supports two structures:
      (a) Split layout:   root/Train/<class>/*.jpg  and  root/Test/<class>/*.jpg
      (b) Flat layout:    root/<class>/*.jpg  (all classes at root, no Train/Test subdirs)
          In flat mode a deterministic stratified train/test split is created via seed/val_ratio.

    Classes used (8-way): AK, BCC, BKL, DF, MEL, NV, SCC, VASC
    UNK is ignored by design.
    """
    CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(self, root, train, transform=None, default_transform=None,
                 only_features=False, method=None, seed=1, val_ratio=0.2):
        self.root = root
        self.transform = transform
        self.no_aug = False
        self.default_transform = default_transform

        train_dir = os.path.join(root, "Train")
        if os.path.isdir(train_dir):
            # Structure (a): pre-split Train/Test dirs
            split_dir = train_dir if train else os.path.join(root, "Test")
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"[ISIC2019] Missing split dir: {split_dir}")
            self.samples = []
            for cls in self.CLASSES:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fn in os.listdir(cls_dir):
                    if fn.lower().endswith(IMG_EXTS):
                        self.samples.append((os.path.join(cls_dir, fn), self.CLASS_TO_IDX[cls]))
        else:
            # Structure (b): flat class dirs, deterministic *per-class*
            # (stratified) split, so every class present on disk contributes
            # to both train and test regardless of class imbalance.
            rng = np.random.RandomState(seed)
            self.samples = []
            for cls in self.CLASSES:
                cls_dir = os.path.join(root, cls)
                if not os.path.isdir(cls_dir):
                    continue
                cls_samples = [
                    (os.path.join(cls_dir, fn), self.CLASS_TO_IDX[cls])
                    for fn in sorted(os.listdir(cls_dir))
                    if fn.lower().endswith(IMG_EXTS)
                ]
                idx = np.arange(len(cls_samples))
                rng.shuffle(idx)
                n_test = int(round(val_ratio * len(cls_samples)))
                test_idx = set(idx[:n_test].tolist())
                self.samples.extend(
                    cls_samples[i] for i in range(len(cls_samples))
                    if (i in test_idx) == (not train)
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"[ISIC2019] No images found under: {root}")

        self.targets = [t for _, t in self.samples]

        self.only_features = only_features
        if only_features:
            self.features = load_features("ISIC2019", train=train, method=method)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        if self.only_features:
            img = self.features[index]
        else:
            img = Image.open(path).convert("RGB")
            if self.no_aug and self.default_transform is not None:
                img = self.default_transform(img)
            elif self.transform is not None:
                img = self.transform(img)

        return img, target
