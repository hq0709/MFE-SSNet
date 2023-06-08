# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : Yiheng
# @brief      : 训练
"""

import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from tools.common_tools import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config.config import cfg
from datasets.Udacity import UdacityDataset
import numpy as np
from tools.model_trainer import ModelTrainer
import math

setup_seed(12345)  # 先固定随机种子
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoDriveModel().to(device)



parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--bs', default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch', type=int, default=None)
parser.add_argument('--data_root_dir', default=r"../../data", type=str,
                    help="path to your dataset")
args = parser.parse_args()

cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

if __name__ == "__main__":
    # step0: config
    train_dir = os.path.join(args.data_root_dir, "train")
    valid_dir = os.path.join(args.data_root_dir, "test")
    check_data_dir(train_dir), check_data_dir(valid_dir)

    # 创建logger
    res_dir = os.path.join(BASE_DIR, "..", "results")
    logger, log_dir = make_logger(res_dir)

    # step1： 数据集
    # 构建MyDataset实例， 构建DataLoder
    train_data = UdacityDataset(root_dir=train_dir, transform=cfg.transforms_train)
    valid_data = UdacityDataset(root_dir=valid_dir, transform=cfg.transforms_train)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers,drop_last = True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.test_bs, num_workers=cfg.workers,drop_last = True)

    model = get_model(cfg, logger)
    model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)


    # step3: 损失函数、优化器
    mse_loss = nn.MSELoss()
    # rmse_loss = math.sqrt(mse_loss)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_init)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # step4: 迭代训练
    # 记录训练所采用的模型、损失函数、优化器、配置参数cfg
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, mse_loss, scheduler, optimizer, model))

    loss_rec = {"train": [], "valid": []}
    best_loss, best_epoch = 0, 0

    for epoch in range(cfg.max_epoch):
        loss_train, loss_train_speed, loss_train_steer = ModelTrainer.train(
            train_loader, model, mse_loss, optimizer, scheduler, epoch, device, cfg, logger)

        loss_valid,loss_valid_speed1, loss_valid_speed2, loss_valid_speed3, loss_valid_speed4,loss_valid_speed5,loss_valid_speed6,loss_valid_speed7,loss_valid_speed8,loss_valid_speed9,loss_valid_speed10,\
            loss_valid_speed11,loss_valid_speed12,loss_valid_speed13,loss_valid_speed14,loss_valid_speed15,loss_valid_speed16,loss_valid_speed17,loss_valid_speed18,loss_valid_speed19,loss_valid_speed20,loss_valid_steer1,\
            loss_valid_steer2,loss_valid_steer3,loss_valid_steer4,loss_valid_steer5,loss_valid_steer6,loss_valid_steer7,loss_valid_steer8,loss_valid_steer9,loss_valid_steer10,loss_valid_steer11,loss_valid_steer12,\
            loss_valid_steer13,loss_valid_steer14,loss_valid_steer15,loss_valid_steer16,loss_valid_steer17,loss_valid_steer18,loss_valid_steer19,loss_valid_steer20= ModelTrainer.valid(\
            valid_loader, model, mse_loss, device)

        logger.info("Epoch[{:0>3}/{:0>3}] Train totalloss:{:.4f}Train speedloss:{:.4f} Train steerloss:{:.4f} Valid loss:{:.4f} Valid speed1loss:{:.4f}Valid speed2loss:{:.4f}Valid speed3loss:{:.4f} "
                    "Valid speed4loss:{:.4f}Valid speed5loss:{:.4f}Valid speed6loss:{:.4f}Valid speed7loss:{:.4f}Valid speed8loss:{:.4f}Valid speed9loss:{:.4f}Valid speed10loss:{:.4f}Valid speed11loss:{:.4f}"
                    "Valid speed12loss:{:.4f}Valid speed13loss:{:.4f}Valid speed14loss:{:.4f}Valid speed15loss:{:.4f}Valid speed16loss:{:.4f}Valid speed17loss:{:.4f}Valid speed18loss:{:.4f}Valid speed19loss:{:.4f}"
                    "Valid speed20loss:{:.4f}Valid steer1loss:{:.4f}Valid steer2loss:{:.4f}Valid steer3loss:{:.4f}Valid steer4loss:{:.4f}Valid steer5loss:{:.4f}Valid steer6loss:{:.4f}Valid steer7loss:{:.4f}"
                    "Valid steer8loss:{:.4f}Valid steer9loss:{:.4f}Valid steer10loss:{:.4f}Valid steer11loss:{:.4f}Valid steer12loss:{:.4f}Valid steer13loss:{:.4f}Valid steer14loss:{:.4f}Valid steer15loss:{:.4f}"
                    "Valid steer16loss:{:.4f}Valid steer17loss:{:.4f}Valid steer18loss:{:.4f}Valid steer19loss:{:.4f}Valid steer20loss:{:.4f}". \
                    format(epoch + 1, cfg.max_epoch, loss_train, loss_train_speed,loss_train_steer,loss_valid,loss_valid_speed1,loss_valid_speed2,loss_valid_speed3, loss_valid_speed4, loss_valid_speed5, loss_valid_speed6, loss_valid_speed7,\
                           loss_valid_speed8,loss_valid_speed9,loss_valid_speed10,loss_valid_speed11,loss_valid_speed12,loss_valid_speed13,loss_valid_speed14,loss_valid_speed15,loss_valid_speed16,loss_valid_speed17,loss_valid_speed18,loss_valid_speed19,loss_valid_speed20,\
                           loss_valid_steer1,loss_valid_steer2,loss_valid_steer3,loss_valid_steer4,loss_valid_steer5,loss_valid_steer6,loss_valid_steer7,loss_valid_steer8,loss_valid_steer9,loss_valid_steer10,loss_valid_steer11,loss_valid_steer12,loss_valid_steer13,\
                           loss_valid_steer14,loss_valid_steer15,loss_valid_steer16,loss_valid_steer17,loss_valid_steer18,loss_valid_steer19,loss_valid_steer20))
        scheduler.step()

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)


        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "loss_valid": loss_valid}
        pkl_name = "checkpoint_{}.pkl".format(epoch)
        path_checkpoint = os.path.join(log_dir, pkl_name)
        torch.save(checkpoint, path_checkpoint)


    logger.info("{} done, best loss: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_loss, best_epoch))
