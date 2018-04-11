# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'
from sklearn import svm
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


# 当使用numpy中的loadtxt函数导入该数据集时，假设数据类型dtype为浮点型，但是很明显数据集的第五列的数据类型是字符串并不是浮点型。
# 因此需要额外做一个工作，即通过loadtxt()函数中的converters参数将第五列通过转换函数映射成浮点类型的数据。
# 首先，我们要写出一个转换函数：
# 定义一个函数，将不同类别标签与数字相对应
def iris_type(s):
    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return class_label[s]

#导入数据
filepath='IRIS_dataset.txt'  # 数据文件路径
data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})
#以上4个参数中分别表示：
#filepath ：文件路径。eg：C:/Dataset/iris.txt。
#dtype=float ：数据类型。eg：float、str等。
#delimiter=',' ：数据以什么分割符号分割。eg：‘，’。
#converters={4:iris_type} ：对某一列数据（第四列）进行某种类型的转换，将数据列与转换函数进行映射的字典。eg：{1:fun}，含义是将第2列对应转换函数进行转换。
#                          converters={4: iris_type}中“4”指的是第5列。

# print(data)
#读入结果示例为：
# [[ 5.1  3.5  1.4  0.2  0. ]
#  [ 4.9  3.   1.4  0.2  0. ]
#  [ 4.7  3.2  1.3  0.2  0. ]
#  [ 4.6  3.1  1.5  0.2  0. ]
#  [ 5.   3.6  1.4  0.2  0. ]]



# '''


#将原始数据集划分成训练集和测试集
X ,y=np.split(data,(4,),axis=1) #np.split 按照列（axis=1）进行分割，从第四列开始往后的作为y 数据，之前的作为X 数据。函数 split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)。
x=X[:,0:2] #在 X中取前两列作为特征（为了后期的可视化画图更加直观，故只取前两列特征值向量进行训练）
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)
# 用train_test_split将数据随机分为训练集和测试集，测试集占总数据的30%（test_size=0.3),random_state是随机数种子
# 参数解释：
# x：train_data：所要划分的样本特征集。
# y：train_target：所要划分的样本结果。
# test_size：样本占比，如果是整数的话就是样本的数量。
# random_state：是随机数的种子。
# （随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
# 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。）

#搭建模型
# classifier=svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
# kernel='rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#decision_function_shape='ovo'： one vs one 分类问题

#开始训练
classifier.fit(x_train,x_test)

#输出训练集的准确率
print(classifier.score(x_train,x_test))

#由于准确率表现不直观，可以通过其他方式观察结果。

#首先将原始结果与训练集预测结果进行对比：
y_train_hat=classifier.predict(x_train)
y_train_1d=y_train.reshape((-1))
comp=zip(y_train_1d,y_train_hat) #用zip把原始结果和预测结果放在一起。显示如下：
print(list(comp))

#同样的,可以用训练好的模型对测试集的数据进行预测:
print(classifier.score(x_test,y_test))
y_test_hat=classifier.predict(x_test)
y_test_1d=y_test.reshape((-1))
comp=zip(y_test_1d,y_test_hat)
print(list(comp))

#还可以通过图像进行可视化：
plt.figure()
plt.subplot(121)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train.reshape((-1)),edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train_hat.reshape((-1)),edgecolors='k',s=50)


# '''
