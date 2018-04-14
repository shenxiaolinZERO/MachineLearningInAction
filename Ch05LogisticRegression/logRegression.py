# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'
# 20180414 Sat

from math import *
import numpy as np
from numpy import *

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
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix * weights ) #h是一个列向量，不是一个数
        error=(labelMat-h)
        weights=weights+alpha * dataMatrix.transpose() * error
    return weights


