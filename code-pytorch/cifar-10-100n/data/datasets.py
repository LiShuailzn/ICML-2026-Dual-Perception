import numpy as np
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
import os



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
def input_dataset(dataset, noise_type, noise_path, is_human, data_root=None):
    # 外部没给就用原来的默认目录
    if data_root is None:
        data_root = os.path.expanduser('~/data')

    # 你本地的实际路径结构：
    # CIFAR-10: /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/cifar-10-batches-py
    # CIFAR-100: /mnt/disk1/lishuai/EA-Dataset/CIFAR-100/cifar-100-python

    if dataset == 'cifar10':
        cifar10_root = os.path.join(data_root, 'CIFAR-10')   # 指到“父目录”
        train_dataset = CIFAR10(
            root=cifar10_root,        # 会在 root/cifar-10-batches-py 下找文件
            download=False,           # 关键：不再下载
            train=True,
            transform=train_cifar10_transform,
            noise_type=noise_type,
            noise_path=noise_path,
            is_human=is_human
        )
        test_dataset = CIFAR10(
            root=cifar10_root,
            download=False,
            train=False,
            transform=test_cifar10_transform,
            noise_type=noise_type
        )
        num_classes = 10
        num_training_samples = 50000

    elif dataset == 'cifar100':
        cifar100_root = os.path.join(data_root, 'CIFAR-100') # 指到“父目录”
        train_dataset = CIFAR100(
            root=cifar100_root,       # 会在 root/cifar-100-python 下找文件
            download=False,           # 关键：不再下载
            train=True,
            transform=train_cifar100_transform,
            noise_type=noise_type,
            noise_path=noise_path,
            is_human=is_human
        )
        test_dataset = CIFAR100(
            root=cifar100_root,
            download=False,
            train=False,
            transform=test_cifar100_transform,
            noise_type=noise_type
        )
        num_classes = 100
        num_training_samples = 50000

    else:
        raise ValueError('dataset must be cifar10 or cifar100')

    return train_dataset, test_dataset, num_classes, num_training_samples








