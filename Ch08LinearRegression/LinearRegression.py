# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

#回归的一般方法：
#（1）收集数据
#（2）准备数据
#（3）分析数据
#（4）训练算法
#（5）测试算法
#（6）使用算法

#标准回归函数和
from numpy import *
# 数据导入函数
#该函数用来打开一个用tab键分隔的文本文件，默认文件每行的最后一个值是目标值。
def loadDataset(fileName):
    numFeat=len(open(fileName).readline().split("\t"))-1
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#standRegres(xArr,yArr)用来计算最佳拟合直线
def standRegres(xArr,yArr):
    #先读入x和y并将它们保存到矩阵中
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    #Numpy 提供一个线性代数的库linalg，其中包括很多有用的函数。
    #可以直接调用linalg.det()来计算行列式
    if linalg.det(xTx)==0.0: #判断xTx的行列式是否为零，如果为零，那么计算逆矩阵的时候将出现错误。所以这一步是必要的！
        print("This matrix is singular, can not do inverse.")
        return
    ws=xTx.I *(xMat.T*yMat)
    #Numpy的线性代数库linalg还提供一个函数来解未知矩阵，如果使用该函数上面一行代码，可替代为：
    #ws=linalg.solve(xTx,xMat.T*yMatT)
    return ws





