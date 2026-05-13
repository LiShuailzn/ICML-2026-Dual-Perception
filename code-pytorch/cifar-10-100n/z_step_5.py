import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from data.datasets import input_dataset
from models import *

FINAL_MODEL_DIR = '/mnt/disk1/lishuai/code-pytorch/step-5'
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='aggre')
parser.add_argument('--noise_path', type=str,
                    help='path of CIFAR-10/100_human.pt', default=None)
parser.add_argument('--dataset', type=str,
                    help='cifar10 or cifar100', default='cifar100')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--data_root', type=str,
                    default='/mnt/disk1/lishuai/EA-Dataset',
                    help='包含 CIFAR-10/CIFAR-100 的根目录')

parser.add_argument('--corrected_labels_path', type=str,
                    default='/mnt/disk1/lishuai/code-pytorch/step-4/corrected_labels.npy',
                    help='最终纠正后的标签 .npy 文件路径')

##################################### Dataset with corrected labels & weights #################


class CorrectedDataset(Dataset):
    def __init__(self, original_dataset, corrected_labels):
        self.original_dataset = original_dataset
        self.corrected_labels = corrected_labels

        print(f'>>> 训练集大小: {len(original_dataset)}')

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, _, orig_index = self.original_dataset[idx]
        label = int(self.corrected_labels[orig_index])
        return img, label, orig_index



##################################### utils ###################################################


def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def train(epoch, train_loader, model, optimizer, args):
    model.train()
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        # 标准交叉熵 loss（不加权）
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1,
                     len(train_loader), train_correct / float(train_total), loss.item()))

    train_acc = float(train_correct) / float(train_total)
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


##################################### main code ################################################

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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

train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human, data_root=args.data_root
)

noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])

if not os.path.exists(args.corrected_labels_path):
    raise FileNotFoundError(f'未找到纠正后标签文件：{args.corrected_labels_path}')
corrected_labels = np.load(args.corrected_labels_path)
assert len(corrected_labels) == num_training_samples, \
    f'corrected_labels 长度 {len(corrected_labels)} != 训练样本数 {num_training_samples}'

corrected_dataset = CorrectedDataset(
    train_dataset,
    corrected_labels,
)

train_loader = DataLoader(dataset=corrected_dataset,
                          batch_size=128,
                          num_workers=args.num_workers,
                          shuffle=True,
                          drop_last=False)


test_loader = DataLoader(dataset=test_dataset,
                         batch_size=64,
                         num_workers=args.num_workers,
                         shuffle=False)

alpha_plan = [0.1] * 60 + [0.01] * 40

# load model
print('building model...')
#model = ResNet18(num_classes)
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
model.cuda()

best_test_acc = 0.0
best_epoch = -1

print('>>> 开始使用纠正标签重新训练模型...')
# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    train_acc = train(epoch, train_loader, model, optimizer, args)
    test_acc = evaluate(test_loader, model)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        save_path = os.path.join(FINAL_MODEL_DIR, 'best_resnet34_corrected_weighted.pth')
        torch.save(model.state_dict(), save_path)
        print(f'>>> new best acc: {best_test_acc:.4f}% (epoch {best_epoch})')
        print(f'>>> the weight is save to: {save_path}')
    else:
        print(f'current acc: {test_acc:.4f}%, best acc: {best_test_acc:.4f}% (epoch {best_epoch})')

    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)

print('>>> 训练完成!')
print(f'>>> 最终最佳测试精度: {best_test_acc:.4f}% (epoch {best_epoch + 1})')
print(f'>>> 最佳模型保存在: {FINAL_MODEL_DIR}')
