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
from matplotlib import colors


def LogRegressionAlgorithm(datas,labels):
    kinds = list(set(labels))  # 3个类别的名字列表
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
    print("W为：",W)

    return W,data


def predict_fun(data,W):
    N, M = data.shape[0], datas.shape[1] + 1  # N是样本数，M是参数向量的维
    K = 3  # k=3是类别数

    # probM每行三个元素，分别表示data中对应样本被判给三个类别的概率
    probM = np.ones((N, K))
    print("data.shape:", data.shape)
    print("datas.shape:", datas.shape)
    print("W.shape:", W.shape)
    print("probM.shape:", probM.shape)

    probM[:, :-1] = np.exp(np.dot(data, W.transpose()))
    probM /= np.array([np.sum(probM, axis=1)]).transpose()  # 得到概率

    predict = np.argmax(probM, axis=1).astype(int)  # 取最大概率对应的类别
    print(predict)

    # #rights 列表储存代表原始标签数据的序号，根据labels 数据生成
    # rights=np.zeros(N)
    # rights[labels == kinds[1]]=1
    # rights[labels == kinds[2]]=2
    # rights =rights.astype(int)

    return predict



    # #误判的个数
    # print("误判的样本的个数为：%d\n"%np.sum(predict !=rights))
    # # print("LR分类器的准确率为：%f\n"%(int(np.sum(predict !=rights)/int(N))))
    # return rights, predict, kinds




if __name__ == '__main__':
    # data_visualization()
    # LogRegressionAlgorithm()
    # rights, predict, kinds=LogRegressionAlgorithm()
    # confusion_matrix(rights, predict, kinds)
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
        # datas.append(linedata[:-1])  # 前4列是4个属性的值(误判的样本的个数为：7
        datas.append(linedata[:-3])  # 前2列是2个属性的值(误判的样本的个数为：30
        labels.append(linedata[-1].replace('\n', ''))  # 最后一列是类别

    datas=np.array(datas)
    datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
    labels=np.array(labels)
    kinds=list(set(labels)) #3个类别的名字列表

    W,data=LogRegressionAlgorithm(datas,labels)
    
    predict=predict_fun(data,W)
    #误判的个数
    # print("误判的样本的个数为：%d\n"%np.sum(predict !=rights))
    # print("LR分类器的准确率为：%f\n"%(int(np.sum(predict !=rights)/int(N))))

    print("predict为：",predict)

    # (5)绘制图像-------------------------------------------------------
    # 1.确定坐标轴范围，x，y轴分别表示两个特征
    x1_min, x1_max = datas[:, 0].min(), datas[:, 0].max()  # 第0列的范围
    x2_min, x2_max = datas[:, 1].min(), datas[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:150j, x2_min:x2_max:150j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    # print("grid_test = \n", grid_test)

    grid_hat = predict_fun(grid_test,W.transpose())  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    # 2.指定默认字体
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 3.绘制
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    alpha = 0.5

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
    # plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
    plt.plot(datas[:, 0], datas[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
    plt.scatter(datas[:, 0], datas[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(u'花萼长度', fontsize=13)
    plt.ylabel(u'花萼宽度', fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    # plt.grid()
    plt.show()
    # -------------------------------------------------------
