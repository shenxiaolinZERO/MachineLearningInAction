# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Xiaolin Shen'
# 20180414 Sat

from math import *
from numpy import *
import matplotlib.pyplot as plt

#打开文本文件并逐行读取
def loadDataSet():
    dataMat=[] ; labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #特征值X0=1.0, X1, X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#sigmoid 函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法函数
def gradAscent(dataMatIn, classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n =shape(dataMatrix)
    alpha=0.001  #目标移动步长
    maxCycles= 500  #迭代次数
    weights=ones((n,1)) #初始化为 n 行1 列的单位矩阵
    for k in range(maxCycles):
        h=sigmoid(dataMatrix * weights ) #h是一个列向量，不是一个数
        error=(labelMat-h)
        weights=weights+alpha * dataMatrix.transpose() * error
    return weights

#随机梯度上升算法函数
def stochasticGradAscent0(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n =shape(dataMatrix)
    alpha=0.01
    weights=ones((n,1)) #初始化为 n 行1 列的单位矩阵
    print("weights.shape:",weights.shape)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=labelMat[i]-h
        weights =weights + alpha * error * dataMatrix[i]
    return weights


#画出数据集合 Logistic回归最佳拟合曲线
def plotBestFit(wei):
    weights=wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]) ; ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]) ; ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr,labelMat=loadDataSet()

    # #---最原始的梯度上升算法：------------------------------------------
    # weights1=gradAscent(dataArr,labelMat)
    # print("gradAscent's weights1：\n",weights1)
    # # [[4.12414349]
    # #  [0.48007329]
    # #  [-0.6168482]]
    # plotBestFit(weights1) #Yes,can plot

    # ---使用随机梯度上升算法：------------------------------------------
    weights2=stochasticGradAscent0(dataArr,labelMat)
    print(weights2)
    plotBestFit(weights2)

