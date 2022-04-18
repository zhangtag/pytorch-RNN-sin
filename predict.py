# 本文件是预测sin函数的main文件
from time import sleep
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import rnn

def predict_sin(net,input,pre_num,input_size):
    """
    通过传入的test_input进行后续pre_num个点的预测，并将预测结果保存下来
    :param net: 使用的模型
    :param test_input: 进行预测的依据，起始数据
    :param index(int): 用户输入的序列，用来分别保存的图片
    :param pre_num(int): 向前预测的点数
    :param input_size:
    """
    delimer = input.size(1)-input_size+1

    with torch.no_grad():
        pre = net.predict(input, pre_num)
        y = pre.cpu().detach().numpy()

    # 画结果
    plt.figure(figsize=(35, 15))
    plt.title("Predict values for sin function", fontsize=25)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)

    colors = ['r','b','c','y','c','m','lime','silver','brown','chartreuse']
    for i in range(input.size(0)):
        # print(len(np.arange(delimer)),len(y[i]))
        # print(len(np.arange(delimer, delimer + pre_num)),len(y[i][delimer:]))
        plt.plot(np.arange(delimer), y[i][:delimer], linewidth=2.0,color=colors[i])
        plt.plot(np.arange(delimer, delimer + pre_num), y[i][delimer:],
                 linewidth=2.0, linestyle=":",color=colors[i])
        #print(y[i][:delimer])

    #plt.show()

    plt.savefig('my_pre_sin_function.jpg')
    #plt.close()


if __name__ == '__main__':
    # 加载数据
    T = 20
    L = 1000
    N = 1 # 
    #plt.ion()
    count = 0
    while count <= 1:

        x = np.empty((N, L), 'int64')
        x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1) 
        data = np.sin(x / 1.0 / T).astype('float64')
        data = torch.from_numpy(data)

        # 通过设置 input_size 决定如何怎样预测：
        # 如果input_size = 3,则使用 n-2,n-1,n -> n+1
        # 如果input_size = 2,则使用 n-1,n -> n+1
        input_size = 3

        # 初始化训练用到的模型,损失函数,优化器

        sinnet = torch.load('predict_sin_func.pkl')
        #sinnet = sinnet
        #sinnet.double()

        pre_num = 100
        predict_sin(sinnet,data,pre_num,input_size)

        count+=1
        sleep(1)
