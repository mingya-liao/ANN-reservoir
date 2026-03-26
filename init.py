# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:33:27 2024
E-mail: 3181780875@qq.com
@author: Dell

"""

import sys
import os
# Get the absolute path to the current script
current_dir = os.getcwd()#获取当前目录
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)#获取当前目录的父目录
# Add the parent directory to the sys.path
sys.path.append(parent_dir+'/supports')#在父目录添加文件夹‘supports’,使用import init即可导入非本文件夹里的函数文件


