import os
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from typing import Any
import pycls.datasets.utils as ds_utils

class ImageNet(datasets.ImageFolder):
    """
    ImageNet Dataset class to handle dataset with annotations under
    `ILSVRC/Annotations/CLS-LOC/train` or `ILSVRC/Annotations/CLS-LOC/val`.

    Args:
        root (string): Root directory of the ImageNet dataset.
        split (string, optional): The dataset split, supports "train", "val", or "test".
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version (e.g., `transforms.RandomCrop`).
        test_transform (callable, optional): Transformations applied during testing.
        only_features (bool, optional): Whether to load precomputed features instead of images.
    """
    def __init__(self, root=None, subset_file=None, split: str = 'train', transform=None, test_transform=None, only_features=False, dataset = None, method=None,  **kwargs: Any):
        self.root = root
        self.test_transform = test_transform
        self.no_aug = False
        self.dataset = dataset
        self.method = method

        # assert self.check_root(), "Something is wrong with the ImageNet dataset path.  {}.".format(
        #     self.root)
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))
        print("subset_file", subset_file)
        # Load WordNet IDs and class mappings



        wnid_to_classes = self.load_wnid_to_classes()
        self.allowed_wnids = self.load_subset_file(subset_file) if subset_file else None
        # print("allowed_wnids", self.allowed_wnids)

        print("split", self.split)



        self.only_features = only_features
        if only_features:
            if split == 'train':
                self.features = ds_utils.load_features(self.dataset, train=True, normalized=False, method=self.method)
            else:
                self.features = ds_utils.load_features(self.dataset, train=False, normalized=False, method=self.method)


        # Remove 'num_classes' from kwargs to avoid issues with ImageFolder
        kwargs.pop("num_classes", None)


        super(ImageNet, self).__init__(self.root, **kwargs)


        if self.allowed_wnids:
            self.wnids = sorted(self.allowed_wnids)
            self.transform = transform
            # self.wnids = self.classes
            # self.wnid_to_idx = self.class_to_idx
            self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
            self.classes = [wnid_to_classes.get(wnid, wnid) for wnid in self.wnids]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            self.filter_dataset()


    def check_root(self):
        """Check if the dataset root directory has the expected structure."""
        required_dirs = [
            os.path.join(self.root, "ILSVRC/Data/CLS-LOC/train"),
            os.path.join(self.root, "ILSVRC/Data/CLS-LOC/val"),
            # os.path.join(self.root, "ILSVRC/Annotations/CLS-LOC/train"),
            # os.path.join(self.root, "ILSVRC/Annotations/CLS-LOC/val"),
        ]
        for d in required_dirs:
            if not os.path.exists(d):
                print(f"Missing directory: {d}")
                return False
        return True


    def load_subset_file(self, subset_file):
        if not os.path.exists(subset_file):
            raise FileNotFoundError(f"Subset file not found: {subset_file}")
        wnids = set()
        with open(subset_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                wnid, class_name = line.strip().split(" ", 1)
                # print("allowed wnid", wnid)
                wnids.add(wnid)
        return wnids

    def filter_dataset(self):
        filtered_samples = []
        filtered_targets = []
        for path, target in self.samples:
            wnid = os.path.basename(os.path.dirname(path))  # Extract class folder name
            if wnid in self.allowed_wnids:
                new_target = self.wnid_to_idx[wnid]
                filtered_samples.append((path, new_target))
                filtered_targets.append(new_target)

        self.samples = filtered_samples
        self.targets = filtered_targets
        print(f"Split: {self.split} | Filtered dataset size: {len(self.samples)}")
        # print(f"First few samples for {self.split}: {[s[0] for s in self.samples[:5]]}")  # Print sample paths for debug

    def load_wnid_to_classes(self):
        """Load mappings from WordNet IDs (wnids) to class names."""
        wnid_to_classes = {}
        mapping_file = '{your_path}/scan/datasets/imagenet/LOC_synset_mapping.txt'  # fill in your local ImageNet root
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found at {mapping_file}. Ensure it exists.")
        with open(mapping_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                wnid, class_name = line.strip().split(" ", 1)
                # print("wnid",wnid)
                wnid_to_classes[wnid] = class_name.strip()
        return wnid_to_classes

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        if self.only_features:
            sample = self.features[index]
        else:
            sample = self.loader(path)
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
        # print(target)
        return sample, target

