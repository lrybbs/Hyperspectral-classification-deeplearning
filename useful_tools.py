# -*- coding: utf-8 -*-
"""
@author RuoyuLiu
@Email lry199801@163.com
@把padding、制作数据集等方法都放在这里了，让主代码显得轻量一点。
@理论上来说我是想实现即便不用我的主代码，这个useful_tools也能发光发热的效果。不过目前来看耦合其实还是很高...日后再改吧。
"""
import numpy as np
import copy
import pandas
import random


# padding函数，先在数据集的上下分别加patch/2行数据(最近的那一行重复加)，再在左右分别加patch/2列数据
def padding(dataset, patch_size):
    padding_dataset = copy.deepcopy(dataset)
    one_side_patch_size = patch_size//2
    up_row = np.repeat([padding_dataset[0, :, :]], one_side_patch_size, axis=0)
    down_row = np.repeat([padding_dataset[-1, :, :]], one_side_patch_size, axis=0)
    padding_dataset = np.concatenate((up_row, padding_dataset, down_row), axis=0)
    # 按列重复完要转置一下，不然拼接不上
    left_column = np.repeat([padding_dataset[:, 0, :]], one_side_patch_size, axis=0).transpose((1, 0, 2))
    right_column = np.repeat([padding_dataset[:, -1, :]], one_side_patch_size, axis=0).transpose((1, 0, 2))
    padding_dataset = np.concatenate((left_column, padding_dataset, right_column), axis=1)
    return padding_dataset


# 制作训练集切片的函数
def make_train_patch(dataset, coordinate, patch_size):
    current_class_train_sample = list()
    # 得到一个个patch_size*patch_size*bands的patch
    for row in coordinate:
        current_class_train_sample.append(dataset[row[0]:row[0]+patch_size, row[1]:row[1]+patch_size, :])
    return np.array(current_class_train_sample)


# 制作训练集的函数
def make_train_set(dataset, ground_truth, sample_num_per_class):
    # 这个标签用来判断是否按比例取训练集
    percent_flag = False
    if sample_num_per_class >= 1.0:
        sample_num_per_class = int(sample_num_per_class)
    else:
        percent_flag = True
    predict_ground_truth = copy.deepcopy(ground_truth)
    # 搜寻每类标签的坐标,随机打乱，取每类前sample_num_per_class个作为训练集
    patch_size = dataset.shape[0] - ground_truth.shape[0] + 1
    label_coordinate = list()
    for i in range(np.max(ground_truth)+1):
        label_coordinate.append(np.array(np.where(ground_truth == i)).T)
    train_sample_coordinate = list()
    if percent_flag:
        for i in range(1, len(label_coordinate)):
            np.random.shuffle(label_coordinate[i])
            train_sample_coordinate.append(label_coordinate[i][0:int(sample_num_per_class * label_coordinate[i].shape[0]), :])
    else:
        for i in range(1, len(label_coordinate)):
            np.random.shuffle(label_coordinate[i])
            train_sample_coordinate.append(label_coordinate[i][0:sample_num_per_class, :])

    # 开始在数据集里面取对应坐标的数据，注意数据集是padding过的
    train_sample, train_label = np.zeros((1, patch_size, patch_size, dataset.shape[2])), np.zeros(1)
    for i in range(len(train_sample_coordinate)):
        train_sample = np.concatenate((train_sample, make_train_patch(dataset, train_sample_coordinate[i], patch_size)), axis=0)
        if percent_flag:
            train_label = np.concatenate((train_label, np.ones(train_sample_coordinate[i].shape[0]) * i), axis=0)
        else:
            train_label = np.concatenate((train_label, np.ones(sample_num_per_class) * i), axis=0)
    train_sample, train_label = train_sample[1:], train_label[1:]
    # 在predict ground truth里面把被取作训练集的数据设为255，这样的话之后算混淆矩阵就可以将其剔除了
    for i in range(len(train_sample_coordinate)):
        for j in range(train_sample_coordinate[i].shape[0]):
            predict_ground_truth[train_sample_coordinate[i][j, 0], train_sample_coordinate[i][j, 1]] = 255
    # 数据增强，竖直、水平、对角线翻转数据集，使其变为原来的4倍
    vertical = np.flip(train_sample, 1)
    horizontal = np.flip(train_sample, 2)
    crosswise = np.flip(horizontal, 2)
    train_sample = np.concatenate((vertical, horizontal, crosswise, train_sample), axis=0)
    train_label = np.concatenate((train_label, train_label, train_label, train_label))
    # 数据增强，高斯加噪
    # noise_train_sample = copy.deepcopy(train_sample)
    # sigma = random.randint(*(10, 15))
    # noise = np.random.normal(0, sigma, size=(train_sample.shape[0], patch_size, patch_size, dataset.shape[2]))
    # noise_train_sample += noise
    # train_sample = np.concatenate((train_sample, noise_train_sample), axis=0)
    # train_label = np.concatenate((train_label, train_label))
    # [N,H,W,C]转[N,C,H,W]，不然Torch读取会出问题
    train_sample = train_sample.transpose((0, 3, 1, 2))
    idx = np.random.permutation(train_sample.shape[0])
    train_sample = train_sample[idx]
    train_label = train_label[idx]
    return predict_ground_truth, train_sample, train_label


# 制作测试集的函数，把整张图都做成测试集，要是不想这么做的话就用上文的办法做就可以了
def get_set_row(dataset, ground_truth, index):
    patch_size = dataset.shape[0] - ground_truth.shape[0] + 1
    test_sample = list()
    test_label = list()
    for j in range(ground_truth.shape[1]):
        test_sample.append(dataset[index:index+patch_size, j:j+patch_size])
        test_label.append(ground_truth[index, j] - 1)
    test_sample, test_label = np.array(test_sample), np.array(test_label)
    test_sample = test_sample.transpose((0, 3, 1, 2))
    return test_sample, test_label


# 这个函数是用来通过混淆矩阵计算总准确率以及分类准确率的，返回的是一个向量
def calculate_acc(confusion_matrix):
    # 加的两项分别是OA和AA，前面的是各类准确率
    acc_vector = np.zeros(confusion_matrix.shape[0] + 2)
    matrix_all = 0
    matrix_correct = 0
    for i in range(confusion_matrix.shape[0]):
        row_all = 0
        row_correct = 0
        for j in range(confusion_matrix.shape[0]):
            matrix_all += confusion_matrix[i, j]
            row_all += confusion_matrix[i, j]
            if i == j:
                matrix_correct += confusion_matrix[i, j]
                row_correct += confusion_matrix[i, j]
        acc_vector[i] = row_correct / row_all
    # 求AA
    acc_vector[-2] = sum(acc_vector) / confusion_matrix.shape[0]
    # 求OA
    acc_vector[-1] = matrix_correct / matrix_all
    return acc_vector


# 这个函数是用来计算f1 score的
def calculate_f1_score(confusion_matrix):
    TP = 0
    FP = 0
    FN = 0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if i == j:
                TP += confusion_matrix[i, j]
            elif i > j:
                FP += confusion_matrix[i, j]
            elif i < j:
                FN += confusion_matrix[i, j]
    recall_mi = TP / (TP + FN)
    precision_mi = TP / (TP + FP)
    micro_f1_score = 2 * (recall_mi * precision_mi) / (recall_mi + precision_mi)
    return micro_f1_score


# 这个类是用来输出详细数据表格的详细数据包括什么训练时间、OA、F1score之类的，还有记录argument里面的一些重要参数。
class OutputData:
    def __init__(self, num_of_label, trial_times):
        self.trial_times = trial_times
        self.params = ('AA', 'OA', 'micro_F1_score', 'train_time', 'predict_time')
        self.params_class = tuple(np.arange(1, num_of_label + 1))
        self.args = ('dataset', 'method', 'patch', 'samples_per_class')
        self.chart = pandas.DataFrame(np.zeros((trial_times + 2, len(self.params + self.params_class))),
                                      columns=self.params_class+self.params, index=np.arange(1, trial_times + 3))
        self.chart = self.chart.rename(index={trial_times + 1: 'average'})
        self.chart = self.chart.rename(index={trial_times + 2: 'arguments'})

    def set_data(self, param_name, current_trail_turn, data):
        self.chart[param_name][current_trail_turn + 1] = data

    # 算一下均值，把argument里面的一些参数写在最后一行，然后就输出
    def output_data(self, path, arguments):
        file_name = path + 'detail_data.xlsx'
        for i in self.params_class:
            self.chart[i]['average'] = self.chart[i][0:self.trial_times].mean()
        for param_name in self.params:
            self.chart[param_name]['average'] = self.chart[param_name][0:self.trial_times].mean()
        for i, arg_name in enumerate(self.args):
            self.chart[self.params[i]]['arguments'] = arg_name + ":" + str(arguments[arg_name])
        self.chart.to_excel(file_name, "detail data")
