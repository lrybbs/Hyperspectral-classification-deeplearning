# -*- coding: utf-8 -*-
"""
@author RuoyuLiu
@Email lry199801@163.com
@部分数据预处理方法参考了https://github.com/sjliu68/MDL4OW，在此鸣谢作者的无私开源。
@pytorchtools.py的作者不是我，来源：https://github.com/Bjarten/early-stopping-pytorch， 实现了pytorch的早停功能。
"""

import os
import useful_tools
import time
import numpy as np
import argparse
from pytorchtools import EarlyStopping
import hyper_net
import torch.utils.data
import matplotlib.pyplot as plt
import spectral

'''
参数设置：
samples_per_class:每类样本数量（默认每类20个）
dataset：选定数据集，默认数据集为Salinas Valley
gt：选定ground truth，正常来说的话肯定是要选和dataset指定的数据集配套的 
trial_turn：实验（训练和测试）进行的轮次，默认进行10次
patch：设定patch的边长，默认大小是9*9
patience：早停之前的轮数，默认是50轮，若样本>=200个每类的话会强制5轮早停
epoch：a是1.0学习率的轮数，b是0.1学习率的轮数，默认值分别是170和30
batch_size：默认是20
verbose：是否输出详细运行信息，默认输出（True）
output：输出文件夹名称设定，默认叫output
'''
parser = argparse.ArgumentParser(description='settings of this tools')
parser.add_argument('--method', type=str, default='HResNet')
parser.add_argument('--dataset', type=str, default='paviaU')
parser.add_argument('--gt', type=str, default='data/paviaU_raw_gt.npy')
parser.add_argument('--trial_turn', type=int, default=5)
parser.add_argument('--samples_per_class', type=float, default=0.02)
parser.add_argument('--patch', type=int, default=9)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--epoch_a', type=int, default=170)
parser.add_argument('--epoch_b', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--output', type=str, default='output_single/')
args = parser.parse_args()
dict_args = vars(args)
# GAN的训练逻辑和普通的网络有所不同，所以如果发现网络是GAN要另做训练逻辑
gan_flag = False
trial_begin_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
# 导入高光谱图片数据dataset和ground truth数据
dataset_path = 'data/' + args.dataset + '_im.npy'
save_path = args.output + args.dataset + '_' + str(round(args.samples_per_class)) \
            + '_per_class_' + trial_begin_time + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
dataset = np.load(dataset_path)
ground_truth = np.load(args.gt)
num_of_label = np.max(ground_truth)
dataset_shape_x, dataset_shape_y, dataset_bands_num = dataset.shape
dataset = np.float32(dataset)
dataset = dataset / dataset.max()
outputs_chart = useful_tools.OutputData(num_of_label, args.trial_turn)
for current_trial_turn in range(args.trial_turn):
    # padding以便进行卷积等操作,得到的padding_dataset在外面加了patch/2圈数据
    padding_dataset = useful_tools.padding(dataset, args.patch)
    # 制作训练集，以及获取一份抠掉训练集的ground truth
    predict_ground_truth, train_set, train_label = \
        useful_tools.make_train_set(padding_dataset, ground_truth, args.samples_per_class)
    train_set = train_set / train_set.max()
    train_set, train_label = torch.as_tensor(torch.from_numpy(train_set), dtype=torch.float32), torch.as_tensor(
        torch.from_numpy(train_label), dtype=torch.long)
    deal_dataset = torch.utils.data.TensorDataset(train_set, train_label)
    train_loader = torch.utils.data.DataLoader(deal_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 网络选择部分
    net = []
    gen_net = []
    if args.method == "HResNet":
        print("Using HResNet")
        net = hyper_net.HResNet(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
    elif args.method == "ResNet-18":
        print("Using ResNet-18")
        net = hyper_net.ResNet18(num_of_bands=dataset_bands_num, num_of_class=num_of_label)
    elif args.method == "2dCNN":
        print("Using 2dCNN")
        net = hyper_net.CNN2d(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
    elif args.method == "FAST3DCNN":
        print("Using FAST3DCNN")
        net = hyper_net.FAST3DCNN(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
    elif args.method == "DCGAN":
        print("Using DCGAN")
        gan_flag = True
        gen_net = hyper_net.DCGenerator(num_of_bands=dataset_bands_num, num_of_class=num_of_label)
        net = hyper_net.DCDiscriminator(num_of_bands=dataset_bands_num, num_of_class=num_of_label)
    else:
        print("the network doesn't exist!")
    # 网络选择部分 end

    # 设定早停，每类样本数大于200时早停默认为patience//10轮
    if args.samples_per_class > 200 or (0.1 < args.samples_per_class < 1.0):
        args.patience = args.patience // 10
    early_stopping = EarlyStopping(args.patience, verbose=args.verbose)

    # 网络训练部分
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_train_begin = time.time()
    net = net.to(device)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0)
    loss_func = torch.nn.CrossEntropyLoss()
    net.train()
    for epoch in range(args.epoch_a):
        loss_list = np.zeros(train_label.shape)
        loss_list_iter = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss_list[loss_list_iter] = loss.cpu().detach().item()
            loss_list_iter += 1
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = np.average(loss_list[0:loss_list_iter])
        early_stopping(avg_loss, net)
        if early_stopping.early_stop:
            break
        if args.verbose:
            print("epoch: " + str(epoch + 1) + " . loss:" + str(round(avg_loss, 6)) +
                  ". accuracy: " + str(round((correct / total)*100, 4)) + ".")
    optimizer = torch.optim.Adadelta(net.parameters(), lr=0.1)
    for epoch in range(args.epoch_b):
        loss_list = np.zeros(train_label.shape)
        loss_list_iter = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss_list[loss_list_iter] = loss.cpu().detach().item()
            loss_list_iter += 1
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = np.average(loss_list[0:loss_list_iter])
        early_stopping(avg_loss, net)
        if early_stopping.early_stop:
            break
        if args.verbose:
            print("epoch: " + str(epoch + 1 + args.epoch_a) + " . loss:" + str(round(avg_loss, 6)) +
                  ". accuracy: " + str(round((correct / total)*100, 4)) + ".")
    time_train_end = time.time()
    print("training end. time:" + str(round(time_train_end - time_train_begin)) + "s")
    outputs_chart.set_data('train_time', current_trial_turn, round(time_train_end - time_train_begin))
    # 网络训练部分 end
    # 预测部分
    # padding_dataset = padding_dataset / padding_dataset.max()
    net.load_state_dict(torch.load("checkpoint.pt"))
    time_predict_begin = time.time()
    net.to(device)
    net = net.eval()
    # 逐个预测、比较，填写混淆矩阵
    predicted_total = np.zeros((dataset_shape_x, dataset_shape_y))
    confusion_matrix = np.zeros((num_of_label, num_of_label))
    with torch.no_grad():
        for i in range(dataset_shape_x):
            test_set_row, test_label_row = useful_tools.get_set_row(padding_dataset, ground_truth, i)
            test_set_row = torch.as_tensor(torch.from_numpy(test_set_row), dtype=torch.float32)
            test_set_row = test_set_row.to(device)
            if gan_flag:
                _, outputs = net(test_set_row)
            else:
                outputs = net(test_set_row)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            # 做点额外说明，ground_truth里面的未知类标号是0，但是预测的时候未知类要排除掉的。而训练的时候label一定要从0开始。
            # 这也就意味着训练集里面的第0类实际应该是gt里的第1类，训练集里面的第1类是gt里的第2类...以此类推
            # 所以为了作出正确预测，要把gt里面的所有值都减一，让gt里的未知类标签变成-1。
            for j in range(predicted.shape[0]):
                if test_label_row[j] != -1:
                    # 为了画出正确图像，应该把预测类（从0开始）全部加1，保证标签形式和gt一致。
                    predicted_total[i, j] = predicted[j] + 1
                    # 若未因作为训练集被剔除，则写入混淆矩阵，进而判断准确率等
                    if test_label_row[j] != 255:
                        confusion_matrix[test_label_row[j], predicted[j]] += 1
                else:
                    predicted_total[i, j] = 0
    acc_vector = useful_tools.calculate_acc(confusion_matrix)
    print("predict_accuracy:" + str(round(acc_vector[-1], 4)))
    time_predict_end = time.time()
    print("predict end. time:" + str(round(time_predict_end - time_predict_begin)) + "s")
    # 将各项数据写入输出表格
    for i in range(1, num_of_label + 1):
        outputs_chart.set_data(i, current_trial_turn, round(acc_vector[i - 1] * 100, 4))
    outputs_chart.set_data('predict_time', current_trial_turn, round(time_predict_end - time_predict_begin))
    outputs_chart.set_data('AA', current_trial_turn, round(acc_vector[-2] * 100, 4))
    outputs_chart.set_data('OA', current_trial_turn, round(acc_vector[-1] * 100, 4))
    outputs_chart.set_data('micro_F1_score', current_trial_turn,
                           round(useful_tools.calculate_f1_score(confusion_matrix) * 100, 4))
    # 预测部分end
    # 画图部分
    ground_predict = spectral.imshow(classes=predicted_total.astype(int))
    plt.axis('off')
    plt.savefig(save_path + str(current_trial_turn + 1) + '.png', dpi=300)
    if args.verbose:
        plt.show()
    plt.close()
    # 画图部分end
    print("turn " + str(current_trial_turn + 1) + " finish!")
# 输出数据和ground truth
outputs_chart.output_data(save_path, dict_args)
ground_truth_print = spectral.imshow(classes=ground_truth.astype(int))
plt.axis('off')
plt.savefig(save_path + 'ground_truth' + '.png', dpi=300)
if args.verbose:
    plt.show()
plt.close()
print("finish!")
