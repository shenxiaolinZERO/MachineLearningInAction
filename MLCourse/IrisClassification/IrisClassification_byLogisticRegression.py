# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'



import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools as it
import matplotlib as mpl


def data_visualization():
    mpl.rcParams['xtick.labelsize']=6
    mpl.rcParams['ytick.labelsize']=6
    mpl.rcParams['axes.labelsize']=6 #设置所有坐标的刻度的字体大小

    attributes=['SepalLength','SepalWidth','PetalLength','PetalWidth'] #鸢尾花的四个属性名
    comb=list(it.combinations([0,1,2,3],3)) #每幅图挑出三个属性绘制三维图

    datas=[]
    labels=[]

    with open('IRIS_dataset.txt','r') as f:
        for line in f:
           linedata=line.split(' ')
           datas.append(linedata[:-1]) #前4列是4个属性的值
           labels.append(linedata[-1].replace('\n','')) #最后一列是类别
    datas=np.array(datas)
    datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
    labels=np.array(labels)
    kinds=list(set(labels)) #3个类别的名字列表

    fig=plt.figure()
    plt.title(u'鸢尾花数据的可视化显示',{'fontname':'STFangsong','fontsize':10})
    for k in range(4):
        ax=fig.add_subplot(int('22'+str(k+1)),projection='3d')
        for label,color,marker in zip(kinds,['r','b','k','y','c','g'],['s','*','o','^']):
            data=datas[labels==label]
            for i in range(data.shape[0]):
                if i==0: #对第一个点标记一个label属性，用于图例的显示，注意label不能设置多次，否则会重复显示
                    ax.plot([data[i,comb[k][0]]],[data[i,comb[k][1]]],[data[i,comb[k][2]]],color=color,marker = marker,markersize=3,label = label,linestyle = 'None')
                else:
                    ax.plot([data[i,comb[k][0]]], [data[i,comb[k][1]]], [data[i,comb[k][2]]], color=color,marker = marker,markersize=3,linestyle = 'None')

        ax.legend(loc="upper left",prop={'size':8},numpoints=1) #图例的放置位置和字体大小

        ax.set_xlabel(attributes[comb[k][0]],fontsize=8) #设置坐标的标签为属性名
        ax.set_ylabel(attributes[comb[k][1]],fontsize=8)
        ax.set_zlabel(attributes[comb[k][2]],fontsize=8)

        ax.set_xticks(np.linspace( np.min(datas[:,comb[k][0]]), np.max(datas[:,comb[k][0]]), 3))
        ax.set_xticks(np.linspace(np.min(datas[:, comb[k][1]]), np.max(datas[:, comb[k][1]]), 3))
        ax.set_xticks(np.linspace(np.min(datas[:, comb[k][2]]), np.max(datas[:, comb[k][3]]), 3))

    plt.subplots_adjust(wspace=0.1,hspace=0.1) #子图之间的空间大小
    # savefig('Iris.png',dpi=700,bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # data_visualization()