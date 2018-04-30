# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

import logRegression as lr
import numpy as np

#使用Logistic回归的随机梯度上升算法来解决病马的生死预测问题
#数据包含368个样本和28个特征。

#步骤：
#（1）收集数据：给定数据文件。
#（2）准备数据：用Python解析文本文件并填充缺失值。
#（3）分析数据：可视化并观察数据。
#（4）训练算法：使用优化算法，找到最佳的系数。
#（5）测试算法：为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练阶段，通过改变迭代的次数和步长等参数来得到更好的回归系数。
#（6）使用算法：实现一个简单的命令行程序来收集马的症状并输出预测结果（not difficult）

#准备数据：处理数据中的缺失值
#（假设有100个样本和20个特征，这些数据都是机器收集回来的。若机器上的某个传感器损坏导致一个特征无效时该怎么办？此时是否要扔掉整个数据？这种情况下，另外19个特征怎么办？它们是否还可用？——答案是肯定的。
# 因为有时候数据相当昂贵，扔掉和重新获取都是不可取的，所以必须采用一些方法来解决这个问题。 ）
#-----一些可选的做法：
#1.使用可用特征的均值来填补缺失值；
#2.使用特殊值来填补缺失值，如-1；
#3.忽略有缺失值的样本；
#4.使用相似样本的均值添补缺失值；
#5.使用另外的机器学习算法预测缺失值。

#在预处理阶段需要做的两件事：
#第一，所有的缺失值必须用一个实数值来替换（因为我们使用的Numpy数据类型不允许包含缺失值）
#     这里选择实数 0 来替换所有缺失值，恰好能适用于Logistic回归。
#     这样做的直觉在于，我们需要的是一个在更新时不会影响系数的值。因为回归系数的更新公式如下：
#     weights=weights+alpha*error*dataMatrix[randIndex]。
#     如果dataMatrix的某特征对应值为0，那么该特征的系数将不做更新，即：weights=weights
#   另外，由于sigmoid（0）=0.5，即它对结果的预测不具有任何倾向性，因此上述做法也不会对误差项造成任何影响。
#   基于上述原因，将缺失值用0代替既可以保留现有数据，也不需要对优化算法进行修改。
#   此外，该数据集中的特征取值一般不为0，因此在某种意义上说它也满足“特殊值”这个要求。
#第二，若在测试数据集中发现了一条数据的类别标签已经缺失，那么我们的简单做法是将该条数据丢弃。这是因为类别标签与特征不同，很难确定采用某个合适的值来替换。
#   采用Logistic回归进行分类时这种做法是合理的，而如果采用类似KNN的方法就可能不太可行。

# 原始的数据集经过预处理之后保存成两个文件：horseColicTest.txt和hourseColicTraining.txt...


#-------------------测试算法：用Logistic回归进行分类
#使用Logistic方法进行分类并不需要做很多工作，所需做的只是把测试集上每个特征向量乘以最优化方法得来的回归系数，再将该乘积结果求和，最后输入到Sigmoid函数中即可。
#如果对应的Sigmoid值大于0.5就预测类别标签为1，否则为0。

#Logistic回归分类函数

# classifyVector()函数，以回归系数和特征向量作为输入来计算对应的Sigmoid值。
# 如果Sigmoid 值大于0.5则函数返回1，否则返回0
def classifyVector(inX,weights):
    prob = lr.sigmoid(sum(inX * weights))
    if prob >0.5 :
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt') #打开训练集
    frTest = open('hourseColicTest.txt')  #打开测试集
    trainingSet = [] ;trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #类别标签是1/0
    #调用stochasticGradAscent1()函数来计算回归系数向量
    trainWeights=lr.stochasticGradAscent1(trainingSet,trainingLabels,500)

    #在系数计算完成之后，导入测试集并计算分类错误率。（整体看来，colicTest()具有完全独立的功能，多次运行得到的结果可能稍有不同，这是因为其中有随机的成分在里面。如果在stochasticGradAscent1()函数中回归系数已经收敛，那么结果才将是确定的）
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount +=1
    errorRate=(float(errorCount)/numTestVec)
    print("The error rate of this test is :%f"%errorRate)
    return errorRate

#multiTest()函数的功能是调用函数colicTest()10次并求结果的平均值。
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum +=colicTest()
    print("After %d iterations, the average error rate is : %f"%(numTests,errorSum/float(numTests)))







#P45
