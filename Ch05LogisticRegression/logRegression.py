# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Xiaolin Shen'
# 20180414 Sat
# 20180418 Wed

from math import *
from numpy import *
import matplotlib.pyplot as plt

#--------------------Logistic 回归的一般过程
#（1）收集数据：采用任意方法收集数据
#（2）准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
#（3）分析数据：采用任意方法对数据进行分析。
#（4）训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
#（5）测试算法：一旦训练步骤完成，分类将会很快。
#（6）使用算法：
# 首先，我们需要输入一些数据，并将其转换成对应的结构化数值；
# 接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；
# 在这之后，我们就可以在输出的类别上做一些其他分析工作。

#--------------------Logistic 回归的优劣
#优点：计算代价不高，易于理解和实现。
#缺点：容易欠拟合，分类精度可能不高。
#适用的数据类型：数值型和标称型数据。

# 为了实现Logistic 回归分类器，我们可以在每个特征上都乘以一个回归系数，然后把所有的结果值相加，将这个总和代入 Sigmoid函数中，
# 进而得到一个范围在0~1之间的数值。任何大于0.5的数据被分入1类，小于0.5即被归入0类。
# 所以，Logistic 回归也可以被看成是一种概率估计。

# Sigmoid函数：sig(z)=1/(1+exp(-z))
# 输入记为z，则z =w0x0+w1x1+w2x2+w3x3+...+wnxn ==>向量的写法可为：z=w^Tx

#打开文本文件并逐行读取
def loadDataSet():
    dataMat=[] ; labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #特征值X0=1.0（为了计算方便）, X1, X2。100×3
        labelMat.append(int(lineArr[2])) #第三个值是数据对应的类别标签。100×1
    return dataMat,labelMat

#sigmoid 函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#--------------------梯度上升算法函数
# 梯度上升算法的伪代码：
# 每个回归系数初始化为1
# 重复R次：
#     计算整个数据集的梯度
#     使用alpha×gradient更新回归系数的向量
#     返回回归系数
def gradAscent(dataMatIn, classLabels):
    #获得输入数据并将它们转换成 Numpy 矩阵
    dataMatrix=mat(dataMatIn) #第一个参数，是一个二维Numpy数组，每列代表每个不同的特征（包含两个特征X1和X2，再加上第0维特征X0，故共有3列），每行代表每个训练样本。故是100×3的矩阵
    labelMat=mat(classLabels).transpose() #第二个参数，是类别标签，是一个1×100的行向量。为了便于计算，需要转置为列向量。
    m,n =shape(dataMatrix)
    alpha=0.001  #目标移动步长
    maxCycles= 500  #迭代次数
    weights=ones((n,1)) #初始化为 n 行1 列的单位矩阵 （n原是特征的个数，这里也就是系数的个数了）
    for k in range(maxCycles):
        h=sigmoid(dataMatrix * weights ) #h是一个元素为一系列浮点数的列向量，不是一个数。列向量的元素个数=样本个数，这里是100。
        # 对应的，运算dataMatrix * weights代表的不止一次乘积计算，，事实上该运算包含了300次的乘积。
        # print("h为：",h)
        error=(labelMat-h) #计算真实类别与预测类别的差值。labelMat元素取值0/1
        # print("error为：",error) #值有正有负。
        weights=weights+alpha * dataMatrix.transpose() * error
    return weights
        # gradAscent's weights1：
        #  [[ 4.12414349]
        #  [ 0.48007329]
        #  [-0.6168482 ]]

# SGA的Background :
# GA在每次更新回归系数时，都需要遍历整个数据集，
# 该方法在处理100个左右的数据集尚可，但如果有数十亿样本和成千上万的特征，计算复杂度太高了。
# 一种改进的方法是：一次仅用一个样本点来更新回归系数，即为SGA。
# 由于可以在新样本到来时对分类器进行增量式更新，因而SGA是一个在线学习算法（VS 一次处理所有数据被称作是“批处理”）。

#随机梯度上升（Stochastic Gradient Ascent）算法函数
#与梯度上升（Gradient Ascent）算法的区别：1、GA是变量h 和误差error都是向量，而SGA则全是数值；
#                                    2、SGA没有矩阵的转换过程，所有变量的数据类型都是Numpy数组。
# 随机梯度上升算法的伪代码：
# 所有回归系数初始化为1
# 对数据集中每个样本：
#     计算该样本的梯度
#     使用alpha×gradient更新回归系数值
# 返回回归系数值
def stochasticGradAscent0(dataMatrix,classLabels):  #拟合出来的直线效果还不错，但不像GA那么完美，这里的分类器错分了三分之一的样本
    # dataMatrix = mat(dataMatIn)
    # labelMat = mat(classLabels).transpose()
    dataMatrix=array(dataMatrix)  #注意这里是数组，不是矩阵
    m,n =shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    print("weights.shape:",weights.shape)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=labelMat[i]-h
        weights =weights + alpha * error * dataMatrix[i]
    return weights
    # 输出
    # weights.shape: (3,)
    # [ 1.01702007  0.85914348 -0.36579921]
    # PDF :P42

#画出数据集合 Logistic回归最佳拟合曲线
def plotBestFit(wei):
    # weights=wei.getA()
    weights=wei
    dataMat,labelMat=loadDataSet() # dataMat：100×3。labelMat：100×1。
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
    y=(-weights[0]-weights[1]*x)/weights[2]  #最佳拟合直线
    # 此处设置了 sigmoid 函数为0，回忆：0是两个分类（类别1和类别0）的分界处。
    # 因此设定 0=w0x0+w1x1+w2x2，然后解出X2和X1的关系式（即分隔线的方程，注意X0=1） ： x2=(-w0-w1x1)/w2

    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr,labelMat=loadDataSet()

    #---最原始的梯度上升算法：------------------------------------------
    weights1=gradAscent(dataArr,labelMat)
    print("gradAscent's weights1：\n",weights1)
    # [[4.12414349]
    #  [0.48007329]
    #  [-0.6168482]]
    plotBestFit(weights1.getA()) #Yes,can plot （注意这里需要一个getA（））

    # ---使用随机梯度上升算法：------------------------------------------
    weights2=stochasticGradAscent0(dataArr,labelMat)
    print(weights2)
    #输出
    # weights.shape: (3,)
    # [ 1.01702007  0.85914348 -0.36579921]
    plotBestFit(weights2)

