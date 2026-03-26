# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:22:15 2024
E-mail: 3181780875@qq.com
@author: Dell

"""

import numpy as np
import pandas as pd
import torch
import copy
import datetime

from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas.plotting import parallel_coordinates
# mpl.use('Agg')
#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams.update({"font.size":14})#此处必须添加此句代码方可改变标题字体大小
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'

import init
from prepare_dataset import prepare_traindata,data_preprocess
from train_sp import cal_nse, cal_metrics
from plot_sp import single_evaluation_plot


def plot_result(args,model,path_t,plot=True):
    '''
    数据完全在函数内制备
    '''
    res_opera=data_preprocess(args)#数据预处理
    train_x,train_y,val_x,val_y,test_x,test_y,train_set,val_set,test_set=prepare_traindata(args,res_opera)

    model.cpu()
    model.eval()

    train_output=model(train_x)
    val_output=model(val_x)
    test_output=model(test_x)#模型预测

    train_set['Q_pred']=train_output[:,0].cpu().detach().numpy()#二维张量转化为一维([:,0])再转化为array，并给train_set最后一列添加径流预测值Q_pred
    val_set['Q_pred']=val_output[:,0].cpu().detach().numpy()
    test_set['Q_pred']=test_output[:,0].cpu().detach().numpy()
    
    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex='row', figsize=(10,8),dpi=200)
        single_evaluation_plot(axes[0], train_set,obs_name='Qt',pred_name='Q_pred',force='It',title='Training Period')
        single_evaluation_plot(axes[1], val_set,obs_name='Qt',pred_name='Q_pred',force='It',title='Validating Period')
        single_evaluation_plot(axes[2], test_set,obs_name='Qt',pred_name='Q_pred',force='It',title='Testing Period')
        plt.tight_layout()
        path=path_t+'/evaluation.png'
        plt.savefig(path,dpi=200, bbox_inches='tight')
        # plt.clf()
        # plt.close()
    return train_set,val_set,test_set


# def group_evaluation_plot(plot_set,fig_name,path_t):
#     fig,ax= plt.subplots(nrows=1, ncols=1, sharex='row', figsize=(10, 4),dpi=200)
#     plt.suptitle(fig_name, fontsize=10, y=0.925, x=0.2)
#     group_single_evaluation_plot(ax, plot_set,'Q_obs', 'Q_pred_mean','Q_pred_std','prcp_mean','Q')
#     path=path_t+f'{fig_name}.png'
#     plt.savefig(path,dpi=200, bbox_inches='tight')
#     plt.clf()
#     plt.close()


# def concat_group_Q(train_list,val_list,test_list,save_col):
#     train_result=pd.concat(train_list,axis=1)
#     train_result.columns = [j + f'_{i}' if train_result.columns.duplicated()[i] else j for i,j in enumerate(train_result.columns)]
#     train_result[['Q_pred_mean', 'Q_pred_std','Q_pred_min','Q_pred_max']] = train_result.filter(like='Q_pred').agg(('mean', 'std', 'min', 'max'), axis=1)
#     train_result=train_result.drop(train_result.columns[range(save_col,train_result.shape[1]-4)], axis=1)

#     val_result=pd.concat(val_list,axis=1)
#     val_result.columns = [j + f'_{i}' if val_result.columns.duplicated()[i] else j for i,j in enumerate(val_result.columns)]
#     val_result[['Q_pred_mean', 'Q_pred_std','Q_pred_min','Q_pred_max']] = val_result.filter(like='Q_pred').agg(('mean', 'std', 'min', 'max'), axis=1)
#     val_result=val_result.drop(val_result.columns[range(save_col,val_result.shape[1]-4)], axis=1)
    
#     test_result=pd.concat(test_list,axis=1)
#     test_result.columns = [j + f'_{i}' if test_result.columns.duplicated()[i] else j for i,j in enumerate(test_result.columns)]
#     test_result[['Q_pred_mean', 'Q_pred_std','Q_pred_min','Q_pred_max']] = test_result.filter(like='Q_pred').agg(('mean', 'std', 'min', 'max'), axis=1)
#     test_result=test_result.drop(test_result.columns[range(save_col,test_result.shape[1]-4)], axis=1)
#     return train_result,val_result,test_result
