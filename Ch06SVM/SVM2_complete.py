# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

# 20180710
# ~
# 2018...


# 这是完整版本
# 简化版本与完整版本两个版本的比较：
# 实现alpha的更改和代数运算的优化环节一模一样。在优化过程中，唯一的不同就是选择alpha的方式。

# Platt SMO 算法是通过一个外循环来选择第一个alpha值的，并且其选择过程会在两种方式之间进行交替：
# 一种方式是在所有数据集上进行单遍扫描，另一种方式则是在非边界alpha中实现单遍扫描。
# 而所谓非边界alpha指的就是那些不等于边界0或C的alpha值。

# 对整个数据集的扫描相当容易，而实现非边界alpha值的扫描时，首先需要建立这些alpha值的列表，然后再对这个表进行遍历。
# 同时，该步骤会跳过那些已知的不会改变的alpha值。

import SVM2_optStruct
import SVM_assistFunc
# 完整Platt SMO算法中的优化例程
def innerL(i,oS):
    Ei = SVM2_optStruct.calcEk(oS,i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
       ((oS.labelMat[i] * Ei >  oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = SVM2_optStruct.selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] -C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H:
                print("L==H")
                return 0
            eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - \
                oS.X[j,:]*oS.X[j,:].T
            if eta >= 0:
                print("eta >=0")
                return 0
            oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
            oS.alphas[j] = SVM_assistFunc.clipAlpha(oS.alphas[j],H,L)
            SVM2_optStruct.updateEk(oS,j)  # 更新误差缓存
            if (abs(oS.alphas[j]-alphaJold) < 0.00001) :
                print("j not moving enough")
                return 0
            oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i] *(alphaJold-oS.alphas[j])
            SVM2_optStruct.updateEk(oS,i)  # 更新误差缓存
            b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T \
                 - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej - oS.labelMat[i] *(oS.alphas[i] -alphaIold)*oS.X[i,:]*oS.X[j,:].T



