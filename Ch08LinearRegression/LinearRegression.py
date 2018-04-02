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

def standRegres(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print("This matrix is singular, can not do inverse.")
        return
    ws=xTx.I *(xMat.T*yMat)
    return ws





