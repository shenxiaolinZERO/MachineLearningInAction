# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'


#鸢尾花数据可视化
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools as it
import matplotlib as mpl

mpl.rcParams['xtick.labelsize']=6
mpl.rcParams['ytick.labelsize']=6
mpl.rcParams['axes.labelsize']=6 #设置所有坐标的刻度的字体大小

attribute=['SepalLength','SepalWidth','PetalLength','PetalWidth'] #鸢尾花的四个属性名
comb=list(it.combinations([0,1,2,3],3)) #每幅图挑出三个属性绘制三维图

datas=[]
labels=[]

with open('IRIS_dataset.txt','r') as f
   for line in f:
       linedata=line.split(' ')
       datas.append(linedata[:-1]) #前4列是4个属性的值
       labels.append(linedata[-1].replace('\n','')) #最后一列是类别
datas=np.array(datas)
datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
labels=np.array(labels)
kinds=list(set(labels)) #3个类别的名字列表

fig=plt.figure()
plt.title(u'鸢尾花数据的可视化显示',{'fontname':})
