# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : YiHeng Zhao
# @date       : 2022-2-10
# @brief      : 模型训练类
"""
import torch
import math
import numpy as np
from collections import Counter

class ModelTrainer(object):
    @staticmethod
    def train(data_loader, model, mse_loss, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        loss_sigma = []
        loss_speed = [] 
        loss_steer = []


        for i, data in enumerate(data_loader):
            image_use, speed_use, speed_label, steer_label = data
            optimizer.zero_grad()
            # 准备输入序列
            img_sequence = torch.stack(image_use).transpose(0, 1).to(device)
            # 前向传播
            speed_use = torch.stack(speed_use)
            speed_use = torch.unsqueeze(speed_use, dim=1)
            speed_use = speed_use.permute(2, 0, 1).float().to(device)
            predictions_speed, predictions_steer = model(img_sequence, speed_use)
            speed_label = torch.stack(speed_label).permute(1,0).to(device)
            steer_label = torch.stack(steer_label).permute(1,0).to(device)
            # k = 20
            # for i in range(0, k):  # loop through k future steps
            #     speed_loss_step = mse_loss(predictions_speed.to(torch.float32)[:, i], speed_label.to(torch.float32)[:, i]) * (1 / i)
            #     speed_loss += speed_loss_step
            # print("predicitions",predictions_speed.shape)
            speed_loss = mse_loss(predictions_speed.to(torch.float32), speed_label.to(torch.float32))
            steer_loss = mse_loss(predictions_steer.to(torch.float32), steer_label.to(torch.float32))
            # 反向传播
            total_loss = speed_loss + steer_loss
            total_loss.backward()
            optimizer.step()
            loss_sigma.append(total_loss.item())
            loss_mean = np.mean(loss_sigma)
            
            loss_speed.append(speed_loss.item())
            loss_mean_speed = np.mean(loss_speed)

            loss_steer.append(steer_loss.item())
            loss_mean_steer = np.mean(loss_steer)

            # 每10个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss_total: {:.4f} Loss_speed: {:.4f} Loss_steer: {:.4f}  ".
                            format(epoch_idx + 1, cfg.max_epoch, i + 1, len(data_loader), loss_mean, speed_loss,steer_loss))
        
        return math.sqrt(loss_mean), math.sqrt(loss_mean_speed), math.sqrt(loss_mean_steer)

    @staticmethod
    def valid(data_loader, model, mse_loss, device):
        model.eval()
        loss_sigma = []
        loss_speed1 = []
        loss_speed2 = []
        loss_speed3 = []
        loss_speed4 = []
        loss_speed5 = []
        loss_speed6 = []
        loss_speed7 = []
        loss_speed8 = []
        loss_speed9 = []
        loss_speed10 = []
        loss_speed11 = []
        loss_speed12 = []
        loss_speed13 = []
        loss_speed14 = []
        loss_speed15 = []
        loss_speed16 = []
        loss_speed17 = []
        loss_speed18 = []
        loss_speed19 = []
        loss_speed20 = []
        loss_steer1 = []
        loss_steer2 = []
        loss_steer3 = []
        loss_steer4 = []
        loss_steer5 = []
        loss_steer6 = []
        loss_steer7 = []
        loss_steer8 = []
        loss_steer9 = []
        loss_steer10 = []
        loss_steer11 = []
        loss_steer12 = []
        loss_steer13 = []
        loss_steer14 = []
        loss_steer15 = []
        loss_steer16 = []
        loss_steer17 = []
        loss_steer18 = []
        loss_steer19 = []
        loss_steer20 = []

        for i, data in enumerate(data_loader):
            image_use, speed_use, speed_label, steer_label = data
            # 准备输入序列
            img_sequence = torch.stack(image_use).transpose(0, 1).to(device)
            # 前向传播
            speed_use = torch.stack(speed_use)
            speed_use = torch.unsqueeze(speed_use, dim=1)
            speed_use = speed_use.permute(2, 0, 1).float().to(device)
            predictions_speed, predictions_steer = model(img_sequence, speed_use)
            speed_label = torch.stack(speed_label).permute(1,0).to(device)
            steer_label = torch.stack(steer_label).permute(1,0).to(device)

            #----------------------------------------------------------------------40个loss
            #speed
            for i in range(0, 20):  # loop through k future steps
                # speed
                speed_loss_step = mse_loss(predictions_speed.to(torch.float32)[:, i], speed_label.to(torch.float32)[:, i])
                if i ==0:
                    lloss_speed1=speed_loss_step
                if i ==1:
                    lloss_speed2=speed_loss_step
                if i ==2:
                    lloss_speed3=speed_loss_step
                if i ==3:
                    lloss_speed4=speed_loss_step
                if i ==4:
                    lloss_speed5=speed_loss_step
                if i ==5:
                    lloss_speed6=speed_loss_step
                if i ==6:
                    lloss_speed7=speed_loss_step
                if i ==7:
                    lloss_speed8=speed_loss_step
                if i ==8:
                    lloss_speed9=speed_loss_step
                if i ==9:
                    lloss_speed10=speed_loss_step
                if i ==10:
                    lloss_speed11=speed_loss_step
                if i ==11:
                    lloss_speed12=speed_loss_step
                if i ==12:
                    lloss_speed13=speed_loss_step
                if i ==13:
                    lloss_speed14=speed_loss_step
                if i ==14:
                    lloss_speed15=speed_loss_step
                if i ==15:
                    lloss_speed16=speed_loss_step
                if i ==16:
                    lloss_speed17=speed_loss_step
                if i ==17:
                    lloss_speed18=speed_loss_step
                if i ==18:
                    lloss_speed19=speed_loss_step
                if i ==19:
                    lloss_speed20=speed_loss_step
                # steer
                steer_loss_step = mse_loss(predictions_steer.to(torch.float32)[:, i], steer_label.to(torch.float32)[:, i])
                if i ==0:
                    lloss_steer1=steer_loss_step
                if i ==1:
                    lloss_steer2=steer_loss_step
                if i ==2:
                    lloss_steer3=steer_loss_step
                if i ==3:
                    lloss_steer4=steer_loss_step
                if i ==4:
                    lloss_steer5=steer_loss_step
                if i ==5:
                    lloss_steer6=steer_loss_step
                if i ==6:
                    lloss_steer7=steer_loss_step
                if i ==7:
                    lloss_steer8=steer_loss_step
                if i ==8:
                    lloss_steer9=steer_loss_step
                if i ==9:
                    lloss_steer10=steer_loss_step
                if i ==10:
                    lloss_steer11=steer_loss_step
                if i ==11:
                    lloss_steer12=steer_loss_step
                if i ==12:
                    lloss_steer13=steer_loss_step
                if i ==13:
                    lloss_steer14=steer_loss_step
                if i ==14:
                    lloss_steer15=steer_loss_step
                if i ==15:
                    lloss_steer16=steer_loss_step
                if i ==16:
                    lloss_steer17=steer_loss_step
                if i ==17:
                    lloss_steer18=steer_loss_step
                if i ==18:
                    lloss_steer19=steer_loss_step
                if i ==19:
                    lloss_steer20=steer_loss_step
            # ----------------------------------------------------------------------xin loss
            total_loss = lloss_speed1+lloss_speed2+lloss_speed3+lloss_speed4+lloss_speed5+lloss_speed6+lloss_speed7+lloss_speed8+lloss_speed9+lloss_speed10+\
                         lloss_speed11+lloss_speed12+lloss_speed13+lloss_speed14+lloss_speed15+lloss_speed16+lloss_speed17+lloss_speed18+lloss_speed19+lloss_speed20+ \
                         lloss_steer1 + lloss_steer2 + lloss_steer3 + lloss_steer4 + lloss_steer5 + lloss_steer6 + lloss_steer7 + lloss_steer8 + lloss_steer9 + lloss_steer10 + \
                         lloss_steer11 + lloss_steer12 + lloss_steer13 + lloss_steer14 + lloss_steer15 + lloss_steer16 + lloss_steer17 + lloss_steer18 + lloss_steer19 + lloss_steer20
            loss_sigma.append(total_loss.item())
            #
            loss_speed1.append(lloss_speed1.item())
            loss_speed2.append(lloss_speed2.item())
            loss_speed3.append(lloss_speed3.item())
            loss_speed4.append(lloss_speed4.item())
            loss_speed5.append(lloss_speed5.item())
            loss_speed6.append(lloss_speed6.item())
            loss_speed7.append(lloss_speed7.item())
            loss_speed8.append(lloss_speed8.item())
            loss_speed9.append(lloss_speed9.item())
            loss_speed10.append(lloss_speed10.item())
            loss_speed11.append(lloss_speed11.item())
            loss_speed12.append(lloss_speed12.item())
            loss_speed13.append(lloss_speed13.item())
            loss_speed14.append(lloss_speed14.item())
            loss_speed15.append(lloss_speed15.item())
            loss_speed16.append(lloss_speed16.item())
            loss_speed17.append(lloss_speed17.item())
            loss_speed18.append(lloss_speed18.item())
            loss_speed19.append(lloss_speed19.item())
            loss_speed20.append(lloss_speed20.item())
            #
            loss_steer1.append(lloss_steer1.item())
            loss_steer2.append(lloss_steer2.item())
            loss_steer3.append(lloss_steer3.item())
            loss_steer4.append(lloss_steer4.item())
            loss_steer5.append(lloss_steer5.item())
            loss_steer6.append(lloss_steer6.item())
            loss_steer7.append(lloss_steer7.item())
            loss_steer8.append(lloss_steer8.item())
            loss_steer9.append(lloss_steer9.item())
            loss_steer10.append(lloss_steer10.item())
            loss_steer11.append(lloss_steer11.item())
            loss_steer12.append(lloss_steer12.item())
            loss_steer13.append(lloss_steer13.item())
            loss_steer14.append(lloss_steer14.item())
            loss_steer15.append(lloss_steer15.item())
            loss_steer16.append(lloss_steer16.item())
            loss_steer17.append(lloss_steer17.item())
            loss_steer18.append(lloss_steer18.item())
            loss_steer19.append(lloss_steer19.item())
            loss_steer20.append(lloss_steer20.item())




        return math.sqrt(np.mean(loss_sigma)), math.sqrt(np.mean(loss_speed1)),math.sqrt(np.mean(loss_speed2)),math.sqrt(np.mean(loss_speed3)),math.sqrt(np.mean(loss_speed4)),math.sqrt(np.mean(loss_speed5)),math.sqrt(np.mean(loss_speed6)),\
        math.sqrt(np.mean(loss_speed7)),math.sqrt(np.mean(loss_speed8)),math.sqrt(np.mean(loss_speed9)),math.sqrt(np.mean(loss_speed10)),math.sqrt(np.mean(loss_speed11)),math.sqrt(np.mean(loss_speed12)),math.sqrt(np.mean(loss_speed13)),math.sqrt(np.mean(loss_speed14)),\
        math.sqrt(np.mean(loss_speed15)),math.sqrt(np.mean(loss_speed16)),math.sqrt(np.mean(loss_speed17)),math.sqrt(np.mean(loss_speed18)),math.sqrt(np.mean(loss_speed19)),math.sqrt(np.mean(loss_speed20)),\
        math.sqrt(np.mean(loss_steer1)),math.sqrt(np.mean(loss_steer2)),math.sqrt(np.mean(loss_steer3)),math.sqrt(np.mean(loss_steer4)),math.sqrt(np.mean(loss_steer5)),math.sqrt(np.mean(loss_steer6)),math.sqrt(np.mean(loss_steer7)), \
        math.sqrt(np.mean(loss_steer8)),math.sqrt(np.mean(loss_steer9)),math.sqrt(np.mean(loss_steer10)),math.sqrt(np.mean(loss_steer11)),math.sqrt(np.mean(loss_steer12)),math.sqrt(np.mean(loss_steer13)),math.sqrt(np.mean(loss_steer14)),\
        math.sqrt(np.mean(loss_steer15)),math.sqrt(np.mean(loss_steer16)),math.sqrt(np.mean(loss_steer17)),math.sqrt(np.mean(loss_steer18)),math.sqrt(np.mean(loss_steer19)),math.sqrt(np.mean(loss_steer20))
    

