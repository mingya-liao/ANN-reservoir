# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:30:52 2024

@author: James
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import PolynomialDecayLR

class Base_Model(nn.Module):
    def __init__(self,warmup_updates,tot_updates,peak_lr,end_lr,power,weight_decay):
        super(Base_Model,self).__init__()#初始化
        self.warmup_updates=warmup_updates
        self.tot_updates=tot_updates
        self.peak_lr=peak_lr
        self.end_lr=end_lr
        self.power=power
        self.weight_decay=weight_decay
    
    def configure_optimizers(self):#优化器
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.peak_lr,weight_decay=self.weight_decay)
        lr_scheduler = PolynomialDecayLR(  #定义了一个学习率调度器，使用多项式衰减算法
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=self.power)
        return optimizer, lr_scheduler
    
    def save(self, PATH):  # You should not revise this  #保存网络参数、优化动量 防止中断重跑
        model_dict = {
            "network": self.state_dict(),
            "optimizer": self.optimizer.state_dict()
            }
        torch.save(model_dict, PATH)

    def load(self, PATH):  # You should not revise this  #接上，中断可以继续load跑
        checkpoint = torch.load(PATH)
        self.load_state_dict(checkpoint["network"])
        # 如果要儲存過程或是中斷訓練後想繼續可以用喔 ^_^
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        




class linear_model(Base_Model):#基于Base_Model，定义线性模型
    def __init__(self,input_channels,hidden_channels,output_channels,dropout,
                warmup_updates,tot_updates,peak_lr,end_lr,power,weight_decay):
        super(linear_model,self).__init__(warmup_updates,tot_updates,peak_lr,end_lr,power,weight_decay)#初始化
        self.layers=nn.Sequential(nn.Linear(input_channels,hidden_channels),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(hidden_channels,output_channels))#线性模型的层数构造
        
    def forward(self,x):#前向传播
        x=self.layers(x)
        return x