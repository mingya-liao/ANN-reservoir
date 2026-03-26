# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:30:52 2024
E-mail: 3181780875@qq.com
@author: Dell

"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import copy
import pickle
import time

import init
from train_sp import save_model,NSE_loss_BF
from plot_utils import plot_result


def test_model(args,model,test_x,test_y,device):#测试模型，输出NSE（越大越好）
    model.eval()
    inputs=Variable(test_x.to(device=device))#将张量移动到GPU或CPU
    labels=Variable(test_y.to(device=device))
    outputs=model(inputs)                    #模型预测结果
    releaseloss_nse=1-NSE_loss_BF(outputs,labels)#NSE，越大越好
    print('final test releasense=',releaseloss_nse.item())


def val_model(args,epoch,model,writer,val_x,val_y,device):#输出验证NSE（越大越好）
    model.eval()
    inputs=Variable(val_x.to(device=device))
    labels=Variable(val_y.to(device=device))
    outputs=model(inputs)
    releaseloss_nse=1-NSE_loss_BF(outputs,labels)
    writer.add_scalar('loss/val_release_nse',releaseloss_nse.item(),epoch)#TensorBoard记录NSE值
    return releaseloss_nse.item()


def train_epoch(args,epoch,model,optimizer,lr_scheduler,writer,stime,
                train_x, train_y, val_x, val_y,
                device,val_release_nse):#仅仅是一个轮次的训练，学习率会更新一次
    model.train()
    optimizer.zero_grad()
    train_nseloss=[]
    #一次全部把数据喂进去容易导致显存不足
    for i in range(0,train_x.shape[0],args.batch_size):#以batch_size为批量进行循环操作
        inputs=Variable(train_x[i:i+args.batch_size].to(device=device))#B_T_F
        labels=Variable(train_y[i:i+args.batch_size].to(device=device))#B_T_F
        outputs=model(inputs)#输出模型预测
        loss_nse=NSE_loss_BF(outputs,labels)#以1-NSE为损失函数
        train_nseloss.append(loss_nse.item())#将损失函数值添加至train_nseloss列表
        optimizer.zero_grad()
        loss_nse.backward()#构建反向传播计算图
        optimizer.step()#按照给定的优化器优化一步
    lr_scheduler.step()#更新学习率
    train_nseloss=np.array(train_nseloss).mean()#求train_nseloss列表平均，即所有小批量损失函数值的平均
    writer.add_scalar('loss/train_nseloss',train_nseloss,epoch)#记录上一步的平均值
    writer.add_scalar('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'],epoch)#记录学习率
    if epoch%10==0:
        etime=time.time()#10的倍数的迭代周期所需时间
        val_release_nse=val_model(args, epoch, model, writer, val_x,val_y, device)#计算验证期纳什效率系数
        if epoch%100==0:
            print('Epoch:{}'.format(epoch+1),
                  'train_nseloss:{:.4f}'.format(train_nseloss),
                  'val_release_nse:{:.4f}'.format(val_release_nse),
                  'cost_time:{:.1f}s'.format(etime-stime))#输出遇100的倍数时的周期数、训练损失值、验证NSE、时间
    return train_nseloss,val_release_nse#返回训练损失值与验证NSE


def run_exp(args,num_epoch,model,optimizer,lr_scheduler,
            train_x, train_y, val_x, val_y,test_x,test_y,
            device,exp,i):
    stime=time.time()
    train_nselosses=[]
    val_release_nse=-np.inf
    bad_count=0
    best_nse=-np.inf
    best_epoch=0
    best_model=None
    flag=1
    path_t='./result/'+exp+f'repeat{i}/'
    writer = SummaryWriter(path_t)
    for epoch in range(num_epoch):
        train_nseloss,val_release_nse=train_epoch(args, epoch, model, optimizer, lr_scheduler, writer, stime,
                                                    train_x, train_y, val_x, val_y,
                                                    device,val_release_nse)
        train_nselosses.append(train_nseloss)#每一个迭代的损失值写入列表
        epoch_model=copy.deepcopy(model.state_dict())#深度复制模型参数
        if np.isnan(train_nseloss):
            flag=0
            return flag#损失值为缺失值，则flag=0
        if train_nseloss<1:
            flag=1#损失值小于1，则flag=1
            if val_release_nse>best_nse:#NSE值大于之前的best_nse(最大值)，则更新best_nse，训练论数，最佳模型状态
                best_nse = val_release_nse
                best_epoch = epoch
                best_model=epoch_model
                bad_count = 0
            else:
                bad_count += 1#NSE没有增大，则记一次bad_count
        else:
            bad_count = 0#损失值大于1，则flag=0
            flag=0
        if bad_count==args.patience:#NSE值一直增大不了，达到500次则break
            print(f'由于验证期的NSE有{args.patience}轮次没有比best nse大而轮数于此中止!best nse=',best_nse)
            break
    if flag==0:
        return flag#flag最后为0同样返回flag值
    else:#若flag!=0，则保存模型参数·输出测试NSE、画图，返回flag值
        save_model(args,best_epoch,epoch,best_model,epoch_model,model,path_t)#model被重新赋值为best_model
        test_model(args,model,test_x,test_y,device)
        plot_result(args,model,path_t)
        return flag

