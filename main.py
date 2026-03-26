# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:30:52 2024

@author: James
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import os

import init
from model import linear_model
from prepare_dataset import data_preprocess,prepare_traindata
from train_utils import run_exp
from utils_sp import set_seed


parser = argparse.ArgumentParser(description='arguments')
#file
parser.add_argument("--data_path", type=str, default='../data/records/', help="data path")
parser.add_argument("--exp", type=str, default='exp1/', help="experiment number")
parser.add_argument("--repeat", type=int, default=0, help="repeat number")
#data setting
parser.add_argument("--train_ratio", type=float, default=0.6, help="training data rato")
parser.add_argument("--val_ratio", type=float, default=0.2, help="validation set data rato")
parser.add_argument("--inflow_shift", type=int, default=1, help="Reservoir ID")
parser.add_argument("--release_shift", type=int, default=1, help="Reservoir ID")
parser.add_argument("--reservoir_ID", type=int, default=41, help="Reservoir ID")
#model setting
parser.add_argument('--model_name', type=str, default='ANN',help='model name')
parser.add_argument('--hidden_channels', type=int, default=8,help='hidden channel size')
parser.add_argument('--output_channels', type=int, default=1,help='output channel size')
parser.add_argument('--dropout', type=float, default=0.,help='dropout probability')
#training setting
parser.add_argument('--seed', type=int, default=42,help='set environment seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--tot_evals', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--tot_updates', type=int, default=2000)
#optimizer setting
parser.add_argument('--warmup_updates', type=int, default=10)
parser.add_argument('--peak_lr', type=float, default=1e-3)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--power', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-3)
args = parser.parse_args()
#%% run the main process
set_seed(args.repeat)#随机种子可复性实现
res_opera=data_preprocess(args)
train_x,train_y,val_x,val_y,test_x,test_y,_,_,_=prepare_traindata(args,res_opera)#数据预处理
[sample_size,input_channel] = train_x.size()#样本数、特征数
args.input_channels = input_channel#给args参数添加一个输入

if args.cuda:
    device = torch.device(f'cuda:{args.cuda_device}')
    print('GPU')#创建一个cuda设备的对象。args.cuda_device指定了要使用的具体GPU设备编号，整数类型
else:
    device=torch.device('cpu')
    print('CPU')#创建一个表示CPU的设备对象


flags=0
while flags<args.tot_evals:
    print("experiment",flags,"/",args.tot_evals)
    model=linear_model(args.input_channels, args.hidden_channels, args.output_channels, 
                         args.dropout,args.warmup_updates, args.tot_updates,
                         args.peak_lr,args.end_lr, args.power, args.weight_decay,)
    optimizer,lr_scheduler=model.configure_optimizers()
    model=model.to(device=device)#将模型数据移动到指定的设备上，进行计算
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)#参数初始化
    num_parameters=sum([p.numel() for p in model.parameters() if p.requires_grad])#计算参数元素数量
    print(f'{args.exp} model {flags} has {num_parameters} learnable parameters')
    exit_flag=run_exp(args, args.tot_updates, model, optimizer, lr_scheduler, 
                        train_x, train_y, val_x, val_y,test_x,test_y, 
                        device, args.exp,flags)
    flags=flags+exit_flag
    if exit_flag==0:
        print('错误终止!')
    else:
        print('迭代结束，没有错误!')


