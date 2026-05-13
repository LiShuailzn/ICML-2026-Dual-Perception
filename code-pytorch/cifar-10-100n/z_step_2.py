# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import DataLoader

from data.datasets import input_dataset
from models import *

from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection

BEST_MODEL_DIR = '/mnt/disk1/lishuai/code-pytorch/step-1-model'
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, 'best_resnet34.pth')

GRAD_ANALYSIS_DIR = '/mnt/disk1/lishuai/code-pytorch/step-2'
os.makedirs(GRAD_ANALYSIS_DIR, exist_ok=True)

Target_Dim = 2048

Trees = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contamination', type=float, default=0.05, help='IForest contamination 比例（k）')
    parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='aggre')
    parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--dataset', type=str, help=' cifar10 or cifar100', default='cifar10')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--is_human', action='store_true', default=False)
    parser.add_argument('--data_root', type=str,
                        default='/mnt/disk1/lishuai/EA-Dataset',
                        help='包含 CIFAR-10/CIFAR-100 的根目录')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='计算梯度时的 batch size，建议保持为 1')

    return parser.parse_args()


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
    model = ResNet34(num_classes=num_classes)
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f'未找到第一步保存的最优模型权重：{BEST_MODEL_PATH}')
    print(f'>>> Loading best ResNet34 weights from: {BEST_MODEL_PATH}')
    state_dict = torch.load(BEST_MODEL_PATH)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    return train_dataset, test_dataset, model, num_classes, num_training_samples


def compute_gradients_for_last_layer(train_dataset, model, args):
    temp_dl = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    grad_dim = model.linear.weight.numel()
    num_samples = len(train_dataset)
    grads = np.zeros((num_samples, grad_dim), dtype=np.float32)

    print(f'>>> 开始计算每个样本的梯度向量: num_samples={num_samples}, grad_dim={grad_dim}')

    # ===== 计时开始 =====
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    idx = 0
    for batch_i, (images, labels, _) in tqdm(enumerate(temp_dl), total=len(temp_dl)):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduce=True)

        first_drv = autograd.grad(loss, model.linear.weight, create_graph=True)[0]
        grads[idx] = first_drv.detach().cpu().numpy().flatten()
        idx += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    gradient_extraction_time = end_time - start_time
    # ===== 计时结束 =====

    assert idx == num_samples, "梯度计算的样本数与训练集大小不一致，请检查 DataLoader 设置！"

    print('>>> 梯度计算完成。')
    print(f'>>> 梯度提取时间消耗: {gradient_extraction_time:.4f} 秒')

    return grads, gradient_extraction_time


def run_isolation_forest(grads, args):
    print('>>> 开始在梯度空间上运行 IsolationForest...')
    print(f'    当前使用的树的数量（n_estimators）= {Trees}')
    print(f'    设定 contamination = {args.contamination:.4f} ({args.contamination * 100:.2f}%)')

    # ===== 计时开始 =====
    start_time = time.perf_counter()

    clf = IsolationForest(
        n_estimators=Trees,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(grads)
    scores = clf.predict(grads)

    end_time = time.perf_counter()
    outlier_detection_time = end_time - start_time
    # ===== 计时结束 =====

    outlier_indices = np.where(scores == -1)[0]
    inlier_indices = np.where(scores == 1)[0]

    print(f'>>> IForest 检测完成：总样本数 {len(grads)}，'
          f'离群样本 {len(outlier_indices)}，非离群样本 {len(inlier_indices)}')
    print(f'>>> 离群检测时间消耗: {outlier_detection_time:.4f} 秒')

    return outlier_indices, inlier_indices, outlier_detection_time

def maybe_project_gradients(grads, args):
    print('.................................................................................')
    n_samples, n_features = grads.shape
    print(f'>>> 检测到数据集为 {args.dataset.upper()}，使用 SparseRandomProjection 对梯度降维后再进行离群检测...')
    print(f'    原始梯度维度: {grads.shape}')
    target_dim = Target_Dim

    projector = SparseRandomProjection(
        n_components=target_dim,
        dense_output=True,
        random_state=args.seed,
    )
    grads_proj = projector.fit_transform(grads).astype(np.float32)
    print(f'    梯度维度: {grads.shape} -> {grads_proj.shape}')
    return grads_proj

def main():
    args = parse_args()
    train_dataset, test_dataset, model, num_classes, num_training_samples = load_datasets_and_model(args)
    grads, gradient_extraction_time = compute_gradients_for_last_layer(train_dataset, model, args)
    grads_for_oda = maybe_project_gradients(grads, args)
    outlier_indices, inlier_indices, outlier_detection_time = run_isolation_forest(grads_for_oda, args)
    harmful_indices = outlier_indices
    np.save(os.path.join(GRAD_ANALYSIS_DIR, 'harmful_outlier_indices.npy'),
            harmful_indices)

    print('>>> 离群样本（有害样本）索引已保存到目录：')
    print(f'    {GRAD_ANALYSIS_DIR}')

    # ===== 汇总打印 =====
    print('\n================ 时间开销统计 ================')
    print(f'梯度提取时间消耗   = {gradient_extraction_time:.4f} 秒')
    print(f'离群检测时间消耗   = {outlier_detection_time:.4f} 秒')
    print(f'两阶段总时间消耗   = {gradient_extraction_time + outlier_detection_time:.4f} 秒')
    print('============================================')


if __name__ == '__main__':
    main()