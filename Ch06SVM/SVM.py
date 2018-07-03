# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Scarlett Shen'
#20180502

# "SVM是最好的现成的分类器。"——这里说的“现成”指的是分类器不加修改即可直接使用。同时，这就意味着在数据上应用基本形式的SVM分类器就可以得到低错误率的结果。
# SVM能够对训练集之外的数据点做出很好的分类决策。

# SVM有很多实现，但本章只关注其中最流行的一种实现，即序列最小优化（Sequential Minimal Optimization，SMO）算法
# 在此之后，将介绍如何使用一种称为核函数（kernel）的方式将SVM扩展到更多数据集上。
# 最后会回顾第1章中手写识别的例子，并考察其能否通过SVM来提高识别的效果。

# 6.1 基于最大间隔分隔数据
#  优点：泛化错误率低，计算开销不大，结果易解释。
#  缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二分类问题。
#  适用数据类型：数值型和标称型数据。


#将数据集分隔开来的直线称为分隔超平面（separating hyperplane）。
#所有数据点都在二维平面上，分隔超平面为：一条直线；
#所给数据集是三维的，分隔超平面为：一个平面；
#若数据集是1024维的，分隔超平面为：一个1023维的某某对象。该对象被称为超平面（hyperplane），也就是分类的决策边界。

#希望能采用这种方式来构建分类器：如果数据点离决策边界越远，那么其最后的预测结果也就越可信。
#支持向量（support vector）就是离分隔超平面最近的那些点。
#接下来 要试着最大化支持向量到分隔面的距离，需要找到此问题的优化求解方法。

# 6.2 寻找最大间隔
# 分隔超平面的形式可以写成：w^Tx+b，向量w和常数b一起描述了所给数据的分隔线或超平面。
# 点A到分隔超平面的距离为：|w^TA+b| / ||w||，这里的常数b类似于Logistic回归中的截距w_0

#  6.2.1 分类器求解的优化问题
# 下面将使用类似海维赛德阶跃函数（即单位阶跃函数）的函数对 w^T+b 作用得到f(w^T+b)，其中当u<0时 f(u)输出-1，反之输出+1.
# 点到分隔面的函数间隔为：label*(w^T+b)
# 点到分隔面的几何间隔为：label*(w^T+b)·1/||w||
# 目标优化函数：...
# 约束条件：...
# ……

#  6.2.2 SVM应用的一般框架
# SVM的一般流程：
# （1）收集数据：可以使用任意方法。
# （2）准备数据：需要数值型数据。
# （3）分析数据：有助于可视化分隔超平面。
# （4）训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
# （5）测试算法：十分简单的计算过程就可以实现。
# （6）使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二分类器，对多类问题应用SVM需要对代码做一些修改。

#  6.3 SMO高效优化算法
# 所需要做的围绕优化的事情就是训练分类器，一旦得到alpha的最优值，我们就得到了分隔超平面（2维平面中就是直线）并能够将之用于数据分类。

#  6.3.1 Platt的SMO算法
#  John Platt 发布了一个称为SMO的强大算法，即序列最小优化（Sequential Minimal Optimization，SMO），用于训练SVM。
#  Platt的SMO算法是将大优化问题分解为多个小优化问题来求解的。这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。在结果完全相同的同时，SMO算法的求解时间短很多。
#  SMO算法的目标是求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量w并得到分隔超平面。
#  SMO算法的工作原理是：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减少另一个。这里所谓的“合适”就是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha必须要在间隔边界之外，而其第二个条件则是这两个alpha该没有就进行过区间化处理或者不在边界上。

#  6.3.2 应用简化版SMO算法处理小规模数据集
#  Platt的SMO算法的完整实现需要大量代码。
#  在此第一个例子中，我们将会对算法进行简化处理，以便了解算法的基本工作思路，之后再基于简化版给出完整版。
#  简化版代码虽然量少但是执行速度慢。
#  Platt的SMO算法中的外循环确定要优化的最佳alpha对，而简化版却会跳过这一部分，首先在数据集上遍历每一个alpha，从而构成alpha对。
#     这里有一点相当重要，就是我们要同时改变两个alpha，之所以这么做是因为我们有一个约束条件：sum\alpha_(i)·label^(i)=0
#     由于改变一个alpha可能会导致该约束条件失效，因此我们总是同时改变两个alpha。
#  为此，我们将构建一个辅助函数，用于在某个区间范围内随机选择一个整数。
#  同时，我们也需要另一个辅助函数，用于在数值太大时对其进行调整。
#            （辅助函数详见：SVM_assistFunc.py）

import math
from  numpy import *
import SVM_assistFunc as svmAss
#接下来在 testSet.txt文件上应用SMO算法。

dataArr,labelArr= svmAss.loadDataSet('testSet.txt')
# print(labelArr)  #[-1.0, -1.0, 1.0, -1.0, ... 采用的类别标签是-1和1，而不是0和1

# SMO函数的伪代码大致如下：
# 1.创建一个alpha向量并将其初始化为0向量
# 2.当迭代次数小于最大迭代次数时（外循环）
#     对数据集中的每个数据向量（内循环）：
#        如果该数据向量可以被优化：
#           随机选择另外一个数据向量
#           同时优化这两个向量
#           如果两个向量都不能被优化，退出内循环
#     如果所有向量都没被优化，增加迭代数目，继续下一次循环

#简化版的SMO算法程序：（最大 第一个程序了！）
#   该函数有5个输入参数，分别是：数据集、类别标签、常数C、容错率和取消前最大的循环次数。
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix= mat(dataMatIn)
    labelMat= mat(classLabels).transpose() #由于转置了类别标签，因此得到的是一个列向量而不是列表！
    b=0
    m,n=shape(dataMatrix)  #得到数据维度
    alphas=mat(zeros((m,1)))  # 元素都为 0 的列矩阵
    iter=0 #该变量存储的是在没有任何alpha 改变的情况下遍历数据集的次数
    while(iter<maxIter):
        alphaPairsChanged = 0 #先设为0，后再对整个集合顺序遍历，此变量用于记录alpha是否已经进行优化。（在循环结束时就会得知这一点）
        for i in range(m):
            # fXi是计算出来的“预测类别”
            fXi = float(multiply(alphas,labelMat).T* (dataMatrix * dataMatrix[i,:].T)) + b
            # Ei 为预测类别与真实类别的误差
            Ei = fXi -float(labelMat[i])
            # 在如下的if语句中，不管是正间隔还是负间隔都会被测试。并且在该if语句中，也同时检查alpha值，以保证其不能同时等于0或C。
            # 由于后面alpha小于0或大于C时将被调整为0或C，所以一旦在该if语句中，它们等于这两个值的话，那么它们就已经在“边界”上了，因而不能再减少或增大。因此也不需要优化了。
            if((labelMat[i] *Ei < -toler) and (alphas[i] <C )) or \
              ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = svmAss.selectJrand(i,m) #随机选择第2个alpha
                fXj =float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) +b # 采用与第一个alpha值相同的误差计算方法
                Ej = fXj- float(labelMat[j])  #计算第2个alpha的误差
                alphaIold = alphas[i].copy() #可将新的alpha值和老的alpha值进行比较
                alphaJold = alphas[j].copy() #需要为alphaIold和alphaJold重新分配新的内存，否则，新旧值比较时，看不到新旧值的变化。
                #这对if-else是为了保证alpha在0和C之间
                if (labelMat[i] != labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if  L==H: #若L和H相等，就不做任何改变，直接continue
                    print("L==H")
                    continue
                # eta 是 alpha[j]的最优修改值
                eta = 2.0 * dataMatrix[i,:] *dataMatrix[j,:].T -\
                        dataMatrix[i,:] * dataMatrix[i,:].T - \
                        dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print("eta >= 0 ")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta #计算出一个新的alpha[j]
                alphas[j] =svmAss.clipAlpha(alphas[j],H,L) #clipAlpha 这是一个辅助函数，它是用于调整大于H或小于L的alpha值。
                if (abs(alphas[j]-alphaJold)<0.00001): # 检查alpha[j]是否有轻微变化，是则退出for循环
                    print("j not moving enough")
                    continue
                # alpha[i]和 alpha[j]同样进行改变，改变大小一样，但改变的方向正好相反。（即，如果一个增加，那么另一个减少）
                alphas[i] +=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #在对alpha[i]和 alpha[j]进行优化之后，给这两个alpha值设置一个常数项 b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                        labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                        labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C >alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C >alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter: %d  i:%d, pairs changed %d "%(iter,i,alphaPairsChanged))

        #在for循环之外，需要检查alpha值是否做了更新，若有则将iter设为0后继续运行程序。
        if (alphaPairsChanged==0):
            iter +=1
        else:
            iter = 0
        print("iteration number : %d "% iter)


    print("b为：",b)
    print("alpha为：",alphas)
    return b,alphas

#P49
#

if __name__ == '__main__':
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)

