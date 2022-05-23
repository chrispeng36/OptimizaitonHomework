# -*- coding: utf-8 -*-
# @Time : 2021/12/30 20:28
# @Author : ChrisPeng
# @FileName: OptimizerOfML.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import math

# 批量梯度下降BGD
# 拟合函数为：y = theta * x
# 代价函数为：J = 1 / (2 * m) * ((theta * x) - y) * ((theta * x) - y).T;
# 梯度迭代为: theta = theta - alpha / m * (x * (theta * x - y).T);
import numpy as np
import matplotlib.pyplot as plt

# 1、单元数据程序
# 以 y=x为例，所以正确的结果应该趋近于theta = 1
def bgd_single():
    # 训练集, 单样本
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 初始化
    m = len(y)
    theta = 0  # 参数
    alpha = 0.01  # 学习率
    threshold = 0.000001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0
    train_lost_list = []
    # 迭代开始
    for i in range(iterations):
        error = 1 / (2 * m) * np.dot(((theta * x) - y).T, ((theta * x) - y))
        # 迭代停止
        train_lost_list.append(error)
        if abs(error) <= threshold:
            break

        theta -= alpha / m * (np.dot(x.T, (theta * x - y)))

    print('单变量：', '迭代次数： %d' % (i + 1), 'theta： %f' % theta,
          'error1： %f' % error)
    return train_lost_list

# 2、多元数据程序
# 以 y=x1+2*x2为例，所以正确的结果应该趋近于theta = [1，2]


def bgd_multi():
    # 训练集，每个样本有2个分量
    x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3,
                                                                           3)])
    y = np.array([3, 5, 6, 5, 7, 10, 8, 9])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0
    train_loss_list = []
    # 迭代开始
    for i in range(iterations):
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        train_loss_list.append(error)
        # 迭代停止
        if abs(error) <= threshold:
            break

        theta -= alpha / m * (np.dot(x.T, (np.dot(x, theta) - y)))

    print('多元变量：', '迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    return train_loss_list

# 多元数据
def sgd():
    # 训练集，每个样本有2个分量
    x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3, 3)])
    y = np.array([3, 5, 6, 5, 7, 10, 8, 9])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0
    train_loss_list = []
    # 迭代开始
    for i in range(iterations):

        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T, (np.dot(x, theta) - y))
        # 迭代停止
        train_loss_list.append(error)
        if abs(error) <= threshold:
            break

        j = np.random.randint(0, m)

        theta -= alpha * (x[j] * (np.dot(x[j], theta) - y[j]))

    print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    return train_loss_list

# 多元数据
def Momentum_sgd():
    # 训练集，每个样本有三个分量
    x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3,
                                                                           3)])
    y = np.array([3, 5, 6, 5, 7, 10, 8, 9])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    momentum = 0.1  # 冲量
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0
    gradient = 0  # 初始梯度为0

    train_loss_list = []
    # 迭代开始
    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        train_loss_list.append(error)
        # 迭代停止
        if abs(error) <= threshold:
            break

        gradient = momentum * gradient + alpha * (x[j] *
                                                  (np.dot(x[j], theta) - y[j]))
        theta -= gradient

    print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    return train_loss_list

def adam():
    # 训练集，每个样本有三个分量
    x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3,
                                                                           3)])
    y = np.array([3, 5, 6, 5, 7, 10, 8, 9])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    momentum = 0.1  # 冲量
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 3000  # 迭代次数
    error = 0  # 初始错误为0

    b1 = 0.9  # 算法作者建议的默认值
    b2 = 0.999  # 算法作者建议的默认值
    e = 0.00000001  #算法作者建议的默认值
    mt = np.zeros(dim)
    vt = np.zeros(dim)
    train_loss_list = []
    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        train_loss_list.append(error)
        if abs(error) <= threshold:
            break

        gradient = x[j] * (np.dot(x[j], theta) - y[j])
        mt = b1 * mt + (1 - b1) * gradient
        vt = b2 * vt + (1 - b2) * (gradient**2)
        mtt = mt / (1 - (b1**(i + 1)))
        vtt = vt / (1 - (b2**(i + 1)))
        vtt_sqrt = np.array([math.sqrt(vtt[0]),
                             math.sqrt(vtt[1])])  # 因为只能对标量进行开方
        theta = theta - alpha * mtt / (vtt_sqrt + e)

    print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    return train_loss_list

if __name__ == '__main__':
    loss_sigle_sgd = bgd_single()

    step1 = np.arange(19)
    plt.plot(step1,loss_sigle_sgd,label='Train_Lost')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.show()



    loss_multi_bgd = bgd_multi()[0:60]
    step2 = range(0,60)

    loss_sgd = sgd()[0:60]
    loss_momentum_sgd = Momentum_sgd()[0:60]
    loss_adam = adam()[0:60]
    # plt.subplot(3,1,1)
    plt.plot(step2,loss_multi_bgd,label='MultiBGD',color='r')
    plt.plot(step2,loss_sgd,label='SGD',color='blue')
    plt.plot(step2,loss_momentum_sgd,label="MomentumSGD",color='green')
    plt.plot(step2,loss_momentum_sgd,label="Adam",color="orange")
    plt.title("Compare different optimizers")
    plt.legend(loc="best")
    plt.show()











