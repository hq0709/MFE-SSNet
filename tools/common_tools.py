# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : Yiheng
# @brief      : 通用函数库
"""
import os
import logging
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from datetime import datetime
from torchvision.models import resnet18
import logging
from models.auto_drive_1 import AutoDriveModel

def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


def check_data_dir(path_tmp):
    assert os.path.exists(path_tmp), \
        "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path_tmp))


def get_model(cfg,logger):
    """
    创建模型
    :param cfg:
    :param cls_num:
    :return:
    """
    if cfg.model_name == "auto_drive_1":
        model = AutoDriveModel()
        logger.info("load auto_drive_1 model!")
    else:
        raise Exception("Invalid model name. got {}".format(cfg.model_name))
    return model


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir

if __name__ == "__main__":

    # setup_seed(2)
    # print(np.random.randint(0, 10, 1))

    logger = Logger('./logtest.log')
    logger = logger.init_logger()
    for i in range(10):
        logger.info('test:' + str(i))

