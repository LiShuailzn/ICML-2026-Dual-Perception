# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.datasets import input_dataset
from models import *

PROCESS_MODEL_DIR = '/mnt/disk1/lishuai/code-pytorch/step-3'
BEST_PURIFIED_MODEL_PATH = os.path.join(PROCESS_MODEL_DIR, 'best_resnet34_purified.pth')

GRAD_ANALYSIS_DIR = '/mnt/disk1/lishuai/code-pytorch/step-2'
HARMFUL_INDICES_PATH = os.path.join(GRAD_ANALYSIS_DIR, f'harmful_outlier_indices.npy')

LABEL_CORRECTION_DIR = '/mnt/disk1/lishuai/code-pytorch/step-4'
os.makedirs(LABEL_CORRECTION_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_type', type=str, default='aggre',
                        help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
    parser.add_argument('--noise_path', type=str, default=None,
                        help='path of CIFAR-10/100 human noise file (.pt)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 或 cifar100')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_human', action='store_true', default=False)
    parser.add_argument('--data_root', type=str,
                        default='/mnt/disk1/lishuai/EA-Dataset',
                        help='包含 CIFAR-10/CIFAR-100 的根目录')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='提取特征时的 batch size')

    return parser.parse_args()


class PurifiedDataset(Dataset):
    def __init__(self, original_dataset, harmful_indices):
        self.original_dataset = original_dataset
        harmful_set = set([int(i) for i in harmful_indices])
        self.purified_indices = [i for i in range(len(original_dataset)) if i not in harmful_set]

    def __len__(self):
        return len(self.purified_indices)

    def __getitem__(self, idx):
        original_idx = self.purified_indices[idx]
        return self.original_dataset[original_idx]


class HarmfulDataset(Dataset):
    def __init__(self, original_dataset, harmful_indices):
        self.original_dataset = original_dataset
        self.harmful_indices = [int(i) for i in harmful_indices]

    def __len__(self):
        return len(self.harmful_indices)

    def __getitem__(self, idx):
        original_idx = self.harmful_indices[idx]
        return self.original_dataset[original_idx]


def load_datasets_and_model(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    noise_type_map = {
        'clean': 'clean_label',
        'worst': 'worse_label',
        'aggre': 'aggre_label',
        'rand1': 'random_label1',
        'rand2': 'random_label2',
        'rand3': 'random_label3',
        'clean100': 'clean_label',
        'noisy100': 'noisy_label',
    }
    if args.noise_type not in noise_type_map:
        raise ValueError(f'未知 noise_type: {args.noise_type}')
    args.noise_type = noise_type_map[args.noise_type]


    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './data/CIFAR-100_human.pt'
        else:
            raise NameError(f'Undefined dataset {args.dataset}')

    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
        args.dataset, args.noise_type, args.noise_path, args.is_human, data_root=args.data_root
    )

    if not os.path.exists(BEST_PURIFIED_MODEL_PATH):
        raise FileNotFoundError(f'未找到净化模型权重：{BEST_PURIFIED_MODEL_PATH}')

    model = ResNet34(num_classes=num_classes)
    print(f'>>> Loading purified best ResNet34 weights from: {BEST_PURIFIED_MODEL_PATH}')
    state_dict = torch.load(BEST_PURIFIED_MODEL_PATH)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return train_dataset, test_dataset, model, num_classes, num_training_samples


def get_feature_extractor_hook(model):
    features_container = {}

    def hook(module, input, output):
        features_container['feat'] = input[0].detach()

    handle = model.linear.register_forward_hook(hook)
    return features_container, handle


def compute_class_prototypes(purified_dataset, model, num_classes, args):
    purified_loader = DataLoader(
        purified_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    feat_container, hook_handle = get_feature_extractor_hook(model)

    prototypes_sum = None
    counts = np.zeros(num_classes, dtype=np.int64)

    print('>>> 开始从净化数据集中提取特征并构建类别原型...')

    with torch.no_grad():
        for images, labels, _ in purified_loader:
            images = images.cuda()
            labels = labels.cuda()

            _ = model(images)
            feats = feat_container['feat']  # [bs, feat_dim]

            if prototypes_sum is None:
                feat_dim = feats.size(1)
                prototypes_sum = np.zeros((num_classes, feat_dim), dtype=np.float32)

            feats_np = feats.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for f, c in zip(feats_np, labels_np):
                prototypes_sum[int(c)] += f
                counts[int(c)] += 1
    hook_handle.remove()
    prototypes = np.zeros_like(prototypes_sum)
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] = prototypes_sum[c] / counts[c]
        else:
            prototypes[c] = np.zeros_like(prototypes_sum[c])

    print('>>> 类别原型构建完成。每类样本数：', counts.tolist())
    return prototypes


def compute_harmful_new_labels(harmful_dataset, model, prototypes, args):
    harmful_loader = DataLoader(
        harmful_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    feat_container, hook_handle = get_feature_extractor_hook(model)
    prototypes_tensor = torch.from_numpy(prototypes).float().cuda()  # [C, D]
    prototypes_norm = prototypes_tensor / (prototypes_tensor.norm(p=2, dim=1, keepdim=True) + 1e-12)

    harmful_new_labels = {}

    print('>>> 开始对有害样本进行特征提取与标签纠正...')

    with torch.no_grad():
        for images, labels, indices in harmful_loader:
            images = images.cuda()
            indices = indices.numpy()

            _ = model(images)
            feats = feat_container['feat']  # [B, D]
            feats_norm = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-12)
            cos_sims = torch.matmul(feats_norm, prototypes_norm.t())
            new_labels_batch = torch.argmax(cos_sims, dim=1).cpu().numpy()

            for idx, new_y in zip(indices, new_labels_batch):
                harmful_new_labels[int(idx)] = int(new_y)

    hook_handle.remove()

    print(f'>>> 有害样本标签纠正完成，共纠正 {len(harmful_new_labels)} 个样本。')
    return harmful_new_labels


def get_original_noisy_labels(train_dataset, num_training_samples, args):
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    original_labels = np.zeros(num_training_samples, dtype=np.int64)

    print('>>> 从原始训练集中恢复噪声标签数组...')

    for images, labels, indices in loader:
        labels_np = labels.numpy()
        indices_np = indices.numpy()
        for idx, y in zip(indices_np, labels_np):
            original_labels[int(idx)] = int(y)

    return original_labels


def main():
    args = parse_args()
    train_dataset, test_dataset, model, num_classes, num_training_samples = load_datasets_and_model(args)
    if not os.path.exists(HARMFUL_INDICES_PATH):
        raise FileNotFoundError(f'未找到有害样本索引文件：{HARMFUL_INDICES_PATH}')
    harmful_indices = np.load(HARMFUL_INDICES_PATH)
    print(f'>>> 加载有害样本索引，共 {len(harmful_indices)} 个。')
    purified_dataset = PurifiedDataset(train_dataset, harmful_indices)
    print(f'>>> 第一次净化数据集大小：{len(purified_dataset)}')
    prototypes = compute_class_prototypes(purified_dataset, model, num_classes, args)
    harmful_dataset = HarmfulDataset(train_dataset, harmful_indices)
    harmful_new_labels = compute_harmful_new_labels(harmful_dataset, model, prototypes, args)
    original_labels = get_original_noisy_labels(train_dataset, num_training_samples, args)

    corrected_labels = original_labels.copy()
    for idx, new_y in harmful_new_labels.items():
        corrected_labels[idx] = new_y
    corrected_labels_path = os.path.join(LABEL_CORRECTION_DIR, 'corrected_labels.npy')
    harmful_new_labels_path = os.path.join(LABEL_CORRECTION_DIR, 'harmful_new_labels_dict.npy')

    np.save(corrected_labels_path, corrected_labels)
    harmful_indices_list = np.array(list(harmful_new_labels.keys()), dtype=np.int64)
    harmful_labels_list = np.array(list(harmful_new_labels.values()), dtype=np.int64)
    np.save(harmful_new_labels_path, np.stack([harmful_indices_list, harmful_labels_list], axis=1))

    print('>>> 标签纠正完成。文件已保存：')
    print(f'    最终纠正标签：{corrected_labels_path}')
    print(f'    有害样本新标签映射：{harmful_new_labels_path}')


if __name__ == '__main__':
    main()
