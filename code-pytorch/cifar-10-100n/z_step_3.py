# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.datasets import input_dataset
from models import *
import argparse

GRAD_ANALYSIS_DIR = '/mnt/disk1/lishuai/code-pytorch/step-2'

PROCESS_MODEL_DIR = '/mnt/disk1/lishuai/code-pytorch/step-3'
os.makedirs(PROCESS_MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='aggre')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--data_root', type=str,
                    default='/mnt/disk1/lishuai/EA-Dataset',
                    help='包含 CIFAR-10/CIFAR-100 的根目录')


class PurifiedDataset(Dataset):
    def __init__(self, original_dataset, harmful_indices):
        self.original_dataset = original_dataset
        self.harmful_indices_set = set(harmful_indices)
        self.purified_indices = [
            i for i in range(len(original_dataset))
            if i not in self.harmful_indices_set
        ]
        print(f'>>> 原始数据集大小: {len(original_dataset)}')
        print(f'>>> 有害样本数量: {len(harmful_indices)}')
        print(f'>>> 净化后数据集大小: {len(self.purified_indices)}')
        print(f'>>> 移除比例: {len(harmful_indices) / len(original_dataset) * 100:.2f}%')

    def __len__(self):
        return len(self.purified_indices)

    def __getitem__(self, idx):
        original_idx = self.purified_indices[idx]
        return self.original_dataset[original_idx]


def load_harmful_indices():
    harmful_indices_path = os.path.join(
        GRAD_ANALYSIS_DIR, 'harmful_outlier_indices.npy'
    )

    if not os.path.exists(harmful_indices_path):
        raise FileNotFoundError(
            f'未找到有害样本索引文件: {harmful_indices_path}\n'
            f'请确认第二步已经运行，并在该目录下生成 harmful_outlier_indices.npy'
        )

    harmful_indices = np.load(harmful_indices_path)
    print(f'>>> 加载有害样本索引: {len(harmful_indices)} 个')
    return harmful_indices




# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_loader), prec, loss.data))  # 修正：使用len(train_loader)


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc



#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print('>>> 步骤1: 加载有害样本索引...')
harmful_indices = load_harmful_indices()
print('>>> 步骤2: 构建净化数据集...')


# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(
    args.dataset,args.noise_type, args.noise_path, args.is_human, data_root=args.data_root
)

# 构建净化数据集（移除有害样本）
purified_train_dataset = PurifiedDataset(train_dataset, harmful_indices)

noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])

train_loader = torch.utils.data.DataLoader(dataset=purified_train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=True,
                                   drop_last = False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)

alpha_plan = [0.1] * 60 + [0.01] * 40

epoch=0
train_acc = 0

# load model
print('building model...')
#model = ResNet18(num_classes)
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
model.cuda()

# 记录历史最好测试精度
best_test_acc = 0.0
best_epoch = -1

# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc = train(epoch, train_loader, model, optimizer)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        save_path = os.path.join(PROCESS_MODEL_DIR, 'best_resnet34_purified.pth')
        torch.save(model.state_dict(), save_path)
        print(f'>>> new best acc: {best_test_acc:.4f}% (epoch {best_epoch})')
        print(f'>>> the weight is save to: {save_path}')
    else:
        print(f'current acc: {test_acc:.4f}%, best acc: {best_test_acc:.4f}% (epoch {best_epoch})')

    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)


print('>>> 训练完成!')
print(f'>>> 最终最佳测试精度: {best_test_acc:.4f}% (epoch {best_epoch + 1})')
print(f'>>> 最佳模型保存在: {PROCESS_MODEL_DIR}')
