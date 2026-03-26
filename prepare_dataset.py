# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:30:52 2024
E-mail: 3181780875@qq.com
@author: Dell

"""

import torch
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import init


def data_preprocess(args):#读取数据、归一化、删掉PDSI、换列名
    res_opera = pd.read_csv(args.data_path+f'{args.reservoir_ID}.csv',index_col=0)
    res_opera['DOY']=res_opera['DOY']/res_opera['DOY'].max()     #归一化
    res_opera.index=pd.to_datetime(res_opera.index)
    res_opera=res_opera[['Storage','NetInflow','Release','DOY']]
    res_opera.columns=['St','It','Qt','Dt']
    return res_opera


def prepare_traindata(args,res_opera):#得到It-1、It+1、Qt-1
    for i in range(args.inflow_shift):
        res_opera[f'It-{i+1}']=res_opera['It'].shift(i+1,axis=0)#下移
        res_opera[f'It+{i+1}']=res_opera['It'].shift(-(i+1),axis=0)#上移
    for i in range(args.release_shift):
        res_opera[f'Qt-{i+1}']=res_opera['Qt'].shift(i+1,axis=0)#下移
    res_opera=res_opera.dropna().copy()
    res_opera['Qt'] = res_opera.pop('Qt')#Qt放最后一列
    time_len=res_opera.shape[0]
    train_end=int(time_len*args.train_ratio)
    val_end=int(time_len*(args.train_ratio+args.val_ratio))
    train_set=res_opera.iloc[:train_end,:]
    val_set=res_opera.iloc[train_end:val_end,:]
    test_set=res_opera.iloc[val_end:,:]#按照比例划分训练集、验证集、测试集,并进一步划分各数据集的特征输入与输出
    train_x,train_y=torch.FloatTensor(train_set.iloc[:,:-1].values),torch.FloatTensor(train_set.iloc[:,-1:].values)
    val_x,val_y=torch.FloatTensor(val_set.iloc[:,:-1].values),torch.FloatTensor(val_set.iloc[:,-1:].values)
    test_x,test_y=torch.FloatTensor(test_set.iloc[:,:-1].values),torch.FloatTensor(test_set.iloc[:,-1:].values)
    return train_x,train_y,val_x,val_y,test_x,test_y,train_set,val_set,test_set



