from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS, default_loader
import os
import torch.nn.functional as F
import random

class MyDataset(VisionDataset):
    def __init__(self, root: str, samples, loader, transform=None, target_transform=None):
        super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.samples = samples
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class TrainValFeed():
    def __init__(self, train_root: str, val_root: str, loader=default_loader, extensions=IMG_EXTENSIONS, train_transform=None, val_transform=None, target_transform=None):
        self.train_root = train_root
        self.val_root = val_root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions

        self.train_samples = self.load_samples(train_root)
        self.val_samples = self.load_samples(val_root)
        
    def load_samples(self, root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples = make_dataset(root, class_to_idx, extensions=self.extensions)
        print(f'{root}: {len(classes)} classes, {len(samples)} samples')
        return samples

    def load(self, subset_ratio: float, seed: int = 8152513, logger = print, _dataset=MyDataset):
        self.subset_ratio = subset_ratio
        logger(f'train {100*subset_ratio:.2f}% of {self.train_root}, {self.val_root}')

        random.seed(seed)
        random.shuffle(self.train_samples)
        train_subset = self.train_samples[:int(len(self.train_samples) * subset_ratio)]
        random.shuffle(self.val_samples)
        val_subset = self.val_samples[:int(len(self.val_samples) * subset_ratio)]
        
        logger(f'train {len(train_subset)}, test {len(val_subset)} samples')

        train_dataset = _dataset(self.train_root, train_subset, self.loader, self.train_transform, self.target_transform)
        val_dataset = _dataset(self.train_root, val_subset, self.loader, self.val_transform, self.target_transform)

        return train_dataset, val_dataset

class MyDataFeed():
    def __init__(self, root: str, loader=default_loader, extensions=IMG_EXTENSIONS, train_transform=None, val_transform=None, target_transform=None):
        self.root = root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions

        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        if len(classes) == 0:
            classes += ['.']
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.samples = make_dataset(root, class_to_idx, extensions=self.extensions)

    def load(self, subset_ratio: float, train_ratio: float, seed: int = 8152513, logger = print, _dataset=MyDataset):
        self.subset_ratio = subset_ratio
        self.train_ratio = train_ratio
        self.val_ratio = 1 - self.train_ratio
        logger(f'train {100*subset_ratio:.2f}% of {self.root}, train-val: {100*self.train_ratio:.2f}%-{100*self.val_ratio:.2f}%')

        random.seed(seed)
        random.shuffle(self.samples)
        subset = self.samples[:int(len(self.samples) * subset_ratio)]
        train_set = subset[:int(len(subset) * train_ratio)]
        val_set = subset[int(len(subset) * train_ratio):]
        logger(f'train {len(train_set)}, test {len(val_set)} samples')

        train_dataset = _dataset(self.root, train_set, self.loader, self.train_transform, self.target_transform)
        val_dataset = _dataset(self.root, val_set, self.loader, self.val_transform, self.target_transform)

        return train_dataset, val_dataset
