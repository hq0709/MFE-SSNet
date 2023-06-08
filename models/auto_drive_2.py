# -*- coding: utf-8 -*-
"""
# @file name  : autodrive_1.py
# @author     : Yiheng
# @brief      : model搭建
"""

import torch
import torch.nn as nn
from torchvision import models
from fusion import AFF

class AutoDriveModel(nn.Module):
    def __init__(self):
        super(AutoDriveModel, self).__init__()

        # 加载预训练的ResNet，如果不重要的话直接用预训练模型接口
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # LSTM层 ：512，因为预训练的 ResNet-18 的最后一个卷积层的输出通道数为 512
        self.lstm_img = nn.LSTM(512, 256, batch_first=True)
        # Timing speed应该只有一层，256为了与图像处理 LSTM 网络的隐藏状态大小保持一致。
        self.lstm_speed = nn.LSTM(1, 256, batch_first=True)

        # 全连接层
        self.fc_speed = nn.Linear(256, 256)
        self.speed_output = nn.Linear(512, 20)
        self.steer_output = nn.Linear(512, 20)

    def forward(self, img_sequence, speed_sequence):
        # ResNet提取图像特征
        # 输入多张连续图像：img_sequence 变量包含了多张连续图像，它们被输入到预训练的 ResNet 模型中进行特征提取
        img_features = [self.resnet(img) for img in img_sequence]
        img_features = torch.stack(img_features).squeeze()

        # LSTM处理
        img_output, _ = self.lstm_img(img_features) #
        speed_output, _ = self.lstm_speed(speed_sequence) #
        # 全连接层
        speed_features = self.fc_speed(speed_output[:, :, :]) # 2 4 256
        # concat 在第 2 个维度（特征维度）上拼接这两个特征。
        fused_features = torch.cat((img_output[:, :, :], speed_features), dim=2)
        # print('speed_features.shape',speed_features.shape)
        # print('fused_features.shape', fused_features.shape)

        aff = AFF(channels=256)
        fused_features = aff(img_output, speed_features)
        print('fused_features.shape', fused_features.shape)
        speed_output = self.speed_output(fused_features)
        steer_output = self.steer_output(fused_features)


        return speed_output, steer_output


if __name__ == '__main__':
        batch_size = 2
        num_frames = 4
        num_channels = 3
        img_height = 224
        img_width = 224

        # 生成随机图像序列和速度序列
        images = [torch.randn(batch_size, num_channels, img_height, img_width) for _ in range(num_frames)]
        speeds = torch.randn(batch_size, num_frames, 1)

        # 将图像序列转换为适合模型输入的格式
        img_sequence = torch.stack(images).transpose(0, 1)
        print("img_sequence",img_sequence.shape)
        print("speeds",speeds.shape)

        # 创建模型实例
        model = AutoDriveModel()

        # 使用随机生成的模拟数据测试网络
        with torch.no_grad():
            speed, steer = model(img_sequence, speeds)

        print(speed.shape, steer.shape)
        print(speed[:, 0, :])
        print(speed[:, -1, :])
