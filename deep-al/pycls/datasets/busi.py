import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class BUSI(Dataset):
    """
    BUSI classification dataset loader (3 classes).

    Expected structure:
      root/
        benign/*.png|jpg...
        malignant/*.png|jpg...
        normal/*.png|jpg...

    We create a deterministic train/test split using (seed, val_ratio).
    'test' here acts like validation/test split for your AL pipeline.
    """
    CLASSES = ["benign", "malignant", "normal"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        default_transform=None,
        seed=1,
        val_ratio=0.2,
        **kwargs
    ):
        self.root = root
        self.transform = transform
        self.default_transform = default_transform
        # compatibility with your training loop toggling augmentation
        self.no_aug = False

        # 1) collect all samples
        all_samples = []
        for cname in self.CLASSES:
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith(IMG_EXTS):
                    all_samples.append((os.path.join(cdir, fn), self.CLASS_TO_IDX[cname]))

        if len(all_samples) == 0:
            raise RuntimeError(
                f"[BUSI] No images found in {root}. Expected class subfolders: {self.CLASSES}"
            )

        # 2) deterministic split
        rng = np.random.RandomState(seed)
        idx = np.arange(len(all_samples))
        rng.shuffle(idx)

        n_test = int(round(val_ratio * len(all_samples)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        use_idx = train_idx if train else test_idx
        self.samples = [all_samples[i] for i in use_idx]

        # 3) targets for your metrics (TV, etc.)
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")

        # match your ISIC behavior: allow no_aug toggle
        if self.no_aug and self.default_transform is not None:
            img = self.default_transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        return img, int(target)
