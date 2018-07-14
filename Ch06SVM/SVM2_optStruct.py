# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

from numpy import *
# 20180711
# ~
# 2018...
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # 误差缓存

    def calcEk(oS,k):
        fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def selectJ(i,oS,Ei):
        # 内循环中的启发式方法
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1,Ei]
        validEcacheList = nonzero(oS.eCache[:,0].A)[0]
        if (len(validEcacheList)) >1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = calcEk(oS,k)
                deletaE = abs(Ei - Ek)
                if (deletaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deletaE
                    Ej = Ek




