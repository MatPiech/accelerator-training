import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision


LABELS_MAPPING = {
    'cifar10': {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'},
    'cifar100': {int(i): i for i in range(100)},
    'flowers': {int(i): i for i in range(102)},
    'food': {int(i): i for i in range(101)},
    'pets': {int(i): i for i in range(37)},
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_loaders(data_path: Path, batch_size: int, norm_mean: tuple[float] = (0.5,), norm_std: tuple[float] = (0.5,), input_shape: tuple[int] = (128, 128)):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_mean, norm_std)
    ])

    if data_path.name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    elif data_path.name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    elif data_path.name == 'flowers':
        trainset = torchvision.datasets.Flowers102(root=data_path, split='train', download=True, transform=transform)
        # valset = torchvision.datasets.Flowers102(root=data_path, split='val', download=True, transform=transform)
        testset = torchvision.datasets.Flowers102(root=data_path, split='test', download=True, transform=transform)
    elif data_path.name == 'food':
        trainset = torchvision.datasets.Food101(root=data_path, split='train', download=True, transform=transform)
        testset = torchvision.datasets.Food101(root=data_path, split='test', download=True, transform=transform)
    elif data_path.name == 'pets':
        trainset = torchvision.datasets.OxfordIIITPet(root=data_path, split='trainval', download=True, transform=transform)
        testset = torchvision.datasets.OxfordIIITPet(root=data_path, split='test', download=True, transform=transform)
    else:
        raise NotImplementedError(f'Dataset {data_path.name} is not supported!')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, num_workers=2)

    return train_loader, test_loader


# Convert logits to predictions
def get_pred(logits):
    return np.argmax(logits, axis=1)


def output_label(label, dataset_name: str) -> str:
    input = (label.item() if type(label) == torch.Tensor else label)
    return LABELS_MAPPING[dataset_name][input]
