# -*- coding: utf-8 -*-
"""
@author RuoyuLiu
@Email lry199801@163.com
@这段说自己是author其实不太合适......这些网络架构有的我是根据人家论文里面的网络图仿真的，有的（比如ResNet-18这种烂大街网
  络）就直接从网上找来改了改参数就放进来了，其实与其说自己是author不如说自己是一个mover。感谢各位作者的无私开源，之后我
  会整理这些网络的思路来源，贴在下面。
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


# 1.HResNet,快速好用
class HResNet(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(HResNet, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.conv0 = nn.Conv2d(self.num_of_bands, 64, kernel_size=(3, 3), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU()
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu12 = nn.ReLU()
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu21 = nn.ReLU()
        self.conv21 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu22 = nn.ReLU()
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.avg_pool = nn.AvgPool2d((patch_size - 2, patch_size - 2))
        # self.dense = nn.Linear(64 * (patch_size - 2) * (patch_size - 2), num_of_class)
        self.dense = nn.Linear(64, num_of_class)

    def forward(self, x):
        x1 = self.conv0(x)
        x0 = x1
        x1 = self.bn1(x1)
        x1 = self.relu11(x1)
        x1 = self.conv11(x1)
        x1 = self.relu12(x1)
        x1 = self.conv12(x1)
        x1 = x0 + x1
        x2 = self.bn2(x1)
        x2 = self.relu21(x2)
        x2 = self.conv21(x2)
        x2 = self.relu22(x2)
        x2 = self.conv22(x2)
        res = x1 + x2
        # res = x0 + x1
        res = self.avg_pool(res)
        res = res.contiguous().view(res.size(0), -1)
        res = self.dense(res)
        return res


# 2.ResNet18及其残差模块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_of_bands, num_of_class):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.num_of_bands, 64, kernel_size=(3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=1)
        self.fc = nn.Linear(512*7*7, self.num_of_class)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = out.contiguous().view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_of_bands, num_of_class):
    return ResNet(ResidualBlock, num_of_bands, num_of_class)


# 3.典中典之2D-CNN
class CNN2d(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(CNN2d, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.conv00 = nn.Conv2d(self.num_of_bands, 64, kernel_size=(3, 3), padding=(0, 0))
        self.conv01 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv02 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu0 = nn.ReLU()
        self.conv10 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 0))
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv20 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(0, 0))
        self.conv21 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv22 = nn.Conv2d(256, 256, kernel_size=(1, 1), padding=(0, 0))
        self.relu2 = nn.ReLU()
        self.avg_pool = nn.AvgPool2d((patch_size - 6, patch_size - 6))
        # self.dense = nn.Linear(256 * (patch_size - 6) * (patch_size - 6), num_of_class)
        self.dense = nn.Linear(256, num_of_class)

    def forward(self, x):
        x0 = self.conv00(x)
        x = self.conv00(x)
        x0 = self.conv01(x0)
        x0 = self.conv02(x0)
        x0 = self.relu0(x0)
        x = x0 + x
        x1 = self.conv10(x)
        x = self.conv10(x)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.relu0(x1)
        x = x1 + x
        x2 = self.conv20(x)
        x = self.conv20(x)
        x2 = self.conv21(x2)
        x2 = self.conv22(x2)
        x2 = self.relu0(x2)
        # out = x2 + x
        res = x2 + x
        out = self.avg_pool(res)
        out = out.contiguous().view(out.size(0), -1)
        out = self.dense(out)
        return out


class FAST3DCNN(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(FAST3DCNN, self).__init__()
        self.patch_size = patch_size
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class

        self.conv1 = nn.Conv3d(1, 8, (7, 3, 3), padding=(0, 0, 0))
        self.conv1_bn = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, (5, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2_bn = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3_bn = nn.BatchNorm3d(32)

        self.conv4 = nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv4_bn = nn.BatchNorm3d(64)

        self.dropout = nn.Dropout(p=0.4)

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.num_of_class)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.num_of_bands, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            b, t, c, w, h = x.size()
        return b * t * c * w * h

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# # 5.DCGAN的生成器和判别器
class DCGenerator(nn.Module):
    # initializers
    def __init__(self, num_of_bands, num_of_class, noise_dim=17):
        super(DCGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class

        self.dense1 = nn.Linear(17, 100)
        self.dense2 = nn.Linear(100, 3 * 3 * 512)

        self.bn1 = nn.BatchNorm2d(momentum=0.8, num_features=512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, (3, 3), (1, 1))
        self.relu1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(momentum=0.8, num_features=256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, (3, 3), (1, 1))
        self.relu2 = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(momentum=0.8, num_features=128)
        self.deconv3 = nn.ConvTranspose2d(128, num_of_bands, (3, 3), (1, 1))
        self.tanh = nn.Tanh()

    # forward method
    def forward(self, inputs):
        x = inputs.contiguous().view(inputs.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.contiguous().view(x.size(0), 512, 3, 3)
        x = self.bn1(x)
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        # x = self.bn3(x)
        x = self.deconv3(x)
        x = self.tanh(x)

        return x


class DCDiscriminator(nn.Module):
    # initializers
    def __init__(self, num_of_bands, num_of_class):
        super(DCDiscriminator, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class

        self.conv1 = nn.Conv2d(num_of_bands, 128, (3, 3), (1, 1))
        self.bn1 = nn.BatchNorm2d(128, momentum=0.8)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(128, 256, (3, 3), (1, 1))
        self.bn2 = nn.BatchNorm2d(256, momentum=0.8)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(256, 512, (3, 3), (1, 1))
        self.bn3 = nn.BatchNorm2d(512, momentum=0.8)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(512, 64, (3, 3), (1, 1))
        self.bn4 = nn.BatchNorm2d(64, momentum=0.8)

        self.dense_bin = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dense_cls = nn.Linear(64, num_of_class)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x.contiguous().view(x.size(0), -1)
        out_bin = self.dense_bin(x)
        out_bin = self.sigmoid(out_bin)
        out_cls = self.dense_cls(x)

        return out_bin, out_cls
