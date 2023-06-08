# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : Yiheng
# @brief      : config
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.transforms as transforms
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.workers = 8
cfg.train_bs = 32
cfg.test_bs = 32
cfg.lr_init = 0.0001
cfg.factor = 0.1
cfg.milestones = [50,125,160]

norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万图像统计得来
norm_std = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normTransform,
])

cfg.log_interval = 10
cfg.model_name = "auto_drive_1"
cfg.max_epoch = 200
