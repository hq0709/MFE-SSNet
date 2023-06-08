# -*- coding: utf-8 -*-
"""
# @file name  : Udacity.py
# @author     : Yiheng
# @brief      : Udacity数据集读取
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class UdacityDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_image_speed, path_label_speed_steer = self.img_info[index]

        image_use = []
        speed_use = []
        for i in range(len(path_image_speed)):
            img = Image.open(path_image_speed[i][0]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            image_use.append(img)
            #   print("speed_steer_test_path", path_image_speed[i])
            with open(path_image_speed[i][1], encoding='utf-8') as file:
                content = file.read()
                content.rstrip()
            speed_use.append(float(content))

        speed_label = []
        steer_label = []
        for i in range(len(path_label_speed_steer)):
            #   print("speed_steer_test_path", path_label_speed_steer[i])
            with open(path_label_speed_steer[i][0], encoding='utf-8') as file:
                content = file.read()
                content.rstrip()
            speed_label.append(float(content))
            with open(path_label_speed_steer[i][1], encoding='utf-8') as file:
                content = file.read()
                content.rstrip()
            steer_label.append(float(content))

        return image_use, speed_use, speed_label, steer_label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        img_info_use = []
        for i in range(len(os.listdir(os.path.join(self.root_dir)))):
            names_image = os.listdir(os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "image"))
            path_images = [os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "image", n) for n in
                           names_image]
            names_speed = os.listdir(os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "speed"))
            path_speeds = [os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "speed", n) for n in
                           names_speed]
            names_steer = os.listdir(os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "steer"))
            path_steers = [os.path.join(self.root_dir, os.listdir(os.path.join(self.root_dir))[i], "steer", n) for n in
                           names_steer]
            # data:数据制作
            # image_speed_use = []
            # for i in range(len(path_images) - 14):  # 最后必须保留15个样本
            #     image_speed_use_item = []
            #     for j in range(10): # 输入前十帧
            #         image_speed = (path_images[i + j], path_speeds[i + j])
            #         image_speed_use_item.append(image_speed)
            #     image_speed_use.append(image_speed_use_item)
            # # print(image_speed_use)
            # # label:数据制作
            # speed_steer_label = []
            # for i in range(10, len(path_images) - 4): # 最后必须保留5个样本
            #     speed_steer_label_item = []
            #     for j in range(5): # 预测5帧
            #         image_speed = (path_speeds[i + j], path_steers[i + j])
            #         speed_steer_label_item.append(image_speed)
            #     speed_steer_label.append(speed_steer_label_item)
            image_speed_use = []
        
            for i in range(len(path_images) - 39):  # 最后必须保留40个样本
                image_speed_use_item = []
                for j in range(20):  # 输入前20帧
                    image_speed = (path_images[i + j], path_speeds[i + j])
                    image_speed_use_item.append(image_speed)
                image_speed_use.append(image_speed_use_item)
            # print(image_speed_use)
            # label:数据制作
            speed_steer_label = []
            for i in range(20, len(path_images) - 19):  # 最后必须保留20个样本
                speed_steer_label_item = []
                for j in range(20):  # 预测20帧
                    image_speed = (path_speeds[i + j], path_steers[i + j])
                    speed_steer_label_item.append(image_speed)
                speed_steer_label.append(speed_steer_label_item)
            img_info = [(p, idx) for p, idx in zip(image_speed_use, speed_steer_label)]
            img_info_use = img_info_use + img_info
        self.img_info = img_info_use
        # print(self.img_info[-1])
