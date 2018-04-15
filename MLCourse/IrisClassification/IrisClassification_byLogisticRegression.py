# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

# 1.Choosing a proper method to classify the IRIS dataset.
# 2.Visualizing raw data and classified data, respectively.
# 3.Submitting with executable codes.
# 4.Submitting a document with the algorithm description and results in detail.
# 5.Submiting to email: XMUML2018@163.com
# 6.The deadline is 2018.04.16.

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
           linedata=line.split(',')
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

def LogRegressionAlgorithm():
    attributes=['SepalLength','SepalWidth','PetalLength','PetalWidth'] #鸢尾花的四个属性名

    datas=[]
    labels=[]

    # with open('IRIS_dataset.txt','r') as f:
    #     for line in f:
    #         linedata=line.split(',')
    #         datas.append(linedata[:-1]) #前4列是4个属性的值
    #         labels.append(linedata[-1].replace('\n','')) #最后一列是类别

    data_file=open('IRIS_dataset.txt','r')
    for line in data_file.readlines():
        # print(line)
        linedata = line.split(',')
        datas.append(linedata[:-1])  # 前4列是4个属性的值
        labels.append(linedata[-1].replace('\n', ''))  # 最后一列是类别

    datas=np.array(datas)
    datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
    labels=np.array(labels)
    kinds=list(set(labels)) #3个类别的名字列表
    # print(datas)
    # print(kinds)
    # print(labels)

    means=datas.mean(axis=0) #各个属性的均值
    stds=datas.std(axis=0) #各个属性的标准差
    N,M= datas.shape[0],datas.shape[1]+1  #N是样本数，M是参数向量的维
    K=3 #k=3是类别数
    data=np.ones((N,M))
    data[:,1:]=(datas-means)/stds #对原始数据进行标准差归一化

    W=np.zeros((K-1,M))  #存储参数矩阵
    priorEs=np.array([1.0/N*np.sum(data[labels==kinds[i]],axis=0) for i in range(K-1)]) #各个属性的先验期望值
    liklist=[]
    for it in range(1000):
        lik=0 #当前的对数似然函数值
        for k in range(K-1): #似然函数值的第一部分
            lik -= np.sum(np.dot(W[k],data[labels==kinds[k]].transpose()))
        lik +=1.0/N *np.sum(np.log(np.sum(np.exp(np.dot(W,data.transpose())),axis=0)+1)) #似然函数的第二部分
        liklist.append(lik)

        wx=np.exp(np.dot(W,data.transpose()))
        probs=np.divide(wx,1+np.sum(wx,axis=0).transpose()) # K-1 *N的矩阵
        posteriorEs=1.0/N*np.dot(probs,data) #各个属性的后验期望值
        gradients=posteriorEs - priorEs +1.0/100 *W #梯度，最后一项是高斯项，防止过拟合
        W -= gradients #对参数进行修正

    #probM每行三个元素，分别表示data中对应样本被判给三个类别的概率
    probM =np.ones((N,K))
    probM[:,:-1]=np.exp(np.dot(data,W.transpose()))
    probM /=np.array([np.sum(probM,axis=1)]).transpose() #得到概率

    predict =np.argmax(probM,axis=1).astype(int) #取最大概率对应的类别

    #rights 列表储存代表原始标签数据的序号，根据labels 数据生成
    rights=np.zeros(N)
    rights[labels == kinds[1]]=1
    rights[labels == kinds[2]]=2
    rights =rights.astype(int)



    #误判的个数
    print("误判的样本的个数为：%d\n"%np.sum(predict !=rights))
    return rights, predict, kinds

#输出混淆矩阵
def confusion_matrix(rights,predict,kinds):
    K=np.shape(list(set(rights).union(set(predict))))[0]
    matrix=np.zeros((K,K))
    for righ,predi in zip(rights,predict):
        matrix[righ,predi] +=1
    print("%-12s"%"类别"),
    for i in range(K):
        print("%-12s"%kinds[i]),
    print('\n'),
    for i in range(K):
        print("%-12s" % kinds[i]),
        for j in range(K):
            print("%-12d"%matrix[i][j]),
        print("\n"),




if __name__ == '__main__':
    # data_visualization()
    # LogRegressionAlgorithm()
    rights, predict, kinds=LogRegressionAlgorithm()
    confusion_matrix(rights, predict, kinds)
