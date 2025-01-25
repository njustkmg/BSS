import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pdb
from dataset.nvGesture import NvGestureDataset
from models.RODModel import ShareClassfier
# from models.basic_model import CLIP_1
from utils.utils import setup_seed, weight_init
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import random
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, accuracy_score

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='/data/php/nvGesture/', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='4', type=str, help='GPU ids')

    return parser.parse_args()

def train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train):

    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    print('Start Training')
    pbar = tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.train()

    for step, bag in enumerate(train_dataloader):
        rgb = bag[0].float().to(device)
        of = bag[1].to(device)
        depth = bag[2].to(device)
        label = bag[3].to(device)

        optimizer.zero_grad()
        # out_mm = net(rgb, of, depth)
        out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

        loss1 = criterion(out_mm, label)
        loss2 = criterion(out_m1, label)
        loss3 = criterion(out_m2, label)
        loss4 = criterion(out_m3, label)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        _total_loss += loss.item()

        pbar.set_postfix(**{'train_loss': _total_loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)

    pbar.close()
    scheduler.step() # 学习率更新

    return _total_loss / epoch_step_train

def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    # 用于收集所有样本的预测结果和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, bag in enumerate(test_dataloader):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            # out_mm = net(rgb, of, depth)
            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            loss1 = criterion(out_mm, label)
            loss2 = criterion(out_m1, label)
            loss3 = criterion(out_m2, label)
            loss4 = criterion(out_m3, label)
            loss = loss1 + loss2 + loss3 + loss4
            _loss += loss.item()

            # out = out_mm
            out = out_m1 + out_m2 + out_m3 + out_mm
            preds = torch.max(out, dim=1)[1]  # 使用out直接获得预测类别
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 注意转换为NumPy数组用于性能计算
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro
def test1(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    # 用于收集所有样本的预测结果和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, bag in enumerate(test_dataloader):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            # out_mm = net(rgb, of, depth)
            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            loss1 = criterion(out_mm, label)
            loss2 = criterion(out_m1, label)
            loss3 = criterion(out_m2, label)
            loss4 = criterion(out_m3, label)
            loss = loss1 + loss2 + loss3 + loss4
            _loss += loss.item()

            # out = out_mm
            out = out_m1
            preds = torch.max(out, dim=1)[1]  # 使用out直接获得预测类别
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 注意转换为NumPy数组用于性能计算
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro
def test2(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    # 用于收集所有样本的预测结果和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, bag in enumerate(test_dataloader):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            # out_mm = net(rgb, of, depth)
            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            loss1 = criterion(out_mm, label)
            loss2 = criterion(out_m1, label)
            loss3 = criterion(out_m2, label)
            loss4 = criterion(out_m3, label)
            loss = loss1 + loss2 + loss3 + loss4
            _loss += loss.item()

            # out = out_mm
            out = out_m2
            preds = torch.max(out, dim=1)[1]  # 使用out直接获得预测类别
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 注意转换为NumPy数组用于性能计算
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro
def test3(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    # 用于收集所有样本的预测结果和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, bag in enumerate(test_dataloader):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            # out_mm = net(rgb, of, depth)
            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            loss1 = criterion(out_mm, label)
            loss2 = criterion(out_m1, label)
            loss3 = criterion(out_m2, label)
            loss4 = criterion(out_m3, label)
            loss = loss1 + loss2 + loss3 + loss4
            _loss += loss.item()

            # out = out_mm
            out = out_m3
            preds = torch.max(out, dim=1)[1]  # 使用out直接获得预测类别
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 注意转换为NumPy数组用于性能计算
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro

def normalize_min_max(values):
    """最小-最大归一化，将输入列表归一化到 [0, 1]"""
    min_val = np.min(values)
    max_val = np.max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def assign_weights(net, train_dataset, device):
    similarities = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不进行 reduction，保留每个样本的 loss
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    net.train()  # 设置模型为训练模式，以便计算梯度

    for step, bag in enumerate(dataloader):
        rgb = bag[0].float().to(device)
        of = bag[1].to(device)
        depth = bag[2].to(device)
        label = bag[3].to(device)

        net.zero_grad()  # 清除前一个批次的梯度
        # out_mm = net(rgb, of, depth)
        out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

        # 计算单模态分类头的余弦相似度
        out_m1_ = F.softmax(out_m1, dim=1)
        out_m2_ = F.softmax(out_m2, dim=1)
        out_m3_ = F.softmax(out_m3, dim=1)
        cosine_sim_m1_m2 = F.cosine_similarity(out_m1_, out_m2_, dim=1)
        cosine_sim_m1_m3 = F.cosine_similarity(out_m1_, out_m3_, dim=1)
        cosine_sim_m2_m3 = F.cosine_similarity(out_m2_, out_m3_, dim=1)
        average_cosine_sim = (cosine_sim_m1_m2 + cosine_sim_m1_m3 + cosine_sim_m2_m3) / 3

        # 使用 detach() 分离计算图并将 Tensor 转为 NumPy 数组
        similarities.extend(average_cosine_sim.detach().cpu().numpy())  # 分离并保存每个样本的余弦相似度

        # 计算每个样本的损失 (批处理情况下)
        loss_m1 = criterion(out_m1, label)  # (shape: [batch_size])
        loss_m2 = criterion(out_m2, label)  # (shape: [batch_size])
        loss_m3 = criterion(out_m3, label)  # (shape: [batch_size])
        loss_mm = criterion(out_mm, label)  # (shape: [batch_size])
        # total_loss = loss_mm  # 对应每个样本的总损失 (shape: [batch_size])
        total_loss = loss_m1 + loss_m2 + loss_m3 + loss_mm  # 对应每个样本的总损失 (shape: [batch_size])
        losses.extend(total_loss.detach().cpu().numpy())  # 分离计算图并保存每个样本的损失

    similarities_norm = normalize_min_max(similarities)
    losses_norm = normalize_min_max(losses) # [0,1]

    combined_scores = [(loss - sim) for sim, loss in zip(similarities_norm, losses_norm)]  # 相似度低 + 损失大 = 更难
    weights = 1 - torch.tensor(combined_scores)
    # weights = 1 - torch.tensor(losses_norm)
    sampling_probs = weights / weights.sum()

    return sampling_probs


if __name__ == '__main__':
    # ---------------------参数----------------------
    args = get_arguments()
    print(args)

    # ---------------------设备----------------------
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print('GPU设备数量为:', torch.cuda.device_count())
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    # ---------------------模型----------------------
    net = ShareClassfier(args)
    net.apply(weight_init)
    net.to(device) # 将模型在指定的device上进行初始化，这里是3号GPU，索引为0号
    net = torch.nn.DataParallel(net, device_ids=gpu_ids) # 对模型进行封装，分发到多个GPU上运行
    net.cuda()

    # ---------------------数据----------------------
    train_dataset = NvGestureDataset(args, mode='train')
    test_dataset = NvGestureDataset(args, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('训练数据量: ', len(train_dataset))
    print('测试数据量: ', len(test_dataset))
    epoch_step_train = len(train_dataset) // train_dataloader.batch_size
    epoch_step_test = math.ceil(len(test_dataset) / test_dataloader.batch_size)  # 因为验证集没有drop_last，所以多一个step，向上取整

    # --------------------优化器---------------------
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    # ------------------训练and验证-------------------
    if True:
        best_acc = 0.0
        best_acc_epoch = 0

        results_file = './results/SPL-both.txt'
        if os.path.exists(results_file):
            os.remove(results_file)

        for epoch in range(args.epochs):
            sampling_probs = assign_weights(net, train_dataset, device)
            sampled_indices = torch.multinomial(sampling_probs, num_samples=len(train_dataset), replacement=False)
            sampled_dataset = torch.utils.data.Subset(train_dataset, sampled_indices)
            current_dataloader = DataLoader(sampled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

            mean_loss_train = train(args, epoch, net, device, current_dataloader, optimizer, scheduler, epoch_step_train)
            mean_loss_test, test_acc, test_f1 = test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test)
            mean_loss_test1, test_acc1, test_f11 = test1(args, epoch, net, device, test_dataloader, optimizer,
                                                     epoch_step_test)
            mean_loss_test2, test_acc2, test_f12 = test2(args, epoch, net, device, test_dataloader, optimizer,
                                                     epoch_step_test)
            mean_loss_test3, test_acc3, test_f13 = test3(args, epoch, net, device, test_dataloader, optimizer,
                                                     epoch_step_test)

            print('********************************************************************')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Now train_loss: %.4f || Now test_loss: %.4f' % (mean_loss_train, mean_loss_test))
            print('Now test_acc: %.4f || Now test_f1: %.4f' % (test_acc, test_f1))

            with open(results_file, 'a') as f:
                f.write(f'{epoch + 1} || {test_acc:.4f} {test_f1:.4f} || {test_acc1:.4f} {test_f11:.4f} || {test_acc2:.4f} {test_f12:.4f} || {test_acc3:.4f} {test_f13:.4f}\n')

            if test_acc > best_acc:
                best_acc = float(test_acc)
                best_acc_epoch = epoch + 1

            print('Best Accuracy: %.4f, Best Epoch: %d' % (best_acc, best_acc_epoch))
            print('********************************************************************')