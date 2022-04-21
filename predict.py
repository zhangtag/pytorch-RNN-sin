# 本文件是预测sin函数的main文件
from os import system
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
    #     print(y[0])

    ax=[]
    ay=[]
    plt.clf()    #清除刷新前的图表，防止数据量过大消耗内存
    #plt.figure(figsize=(35, 15))
    plt.title("Predict values for sin function", fontsize=15)
    #plt.xlabel('x', fontsize=25)
    #plt.ylabel('y', fontsize=25)
    colors = ['r','b','c','y','c','m','lime','silver','brown','chartreuse']
   
    for i in range(input.size(0)):
        # print(len(np.arange(delimer)),len(y[i]))WW
        # print(len(np.arange(delimer, delimer + pre_num)),len(y[i][delimer:]))
        for j in range(delimer + pre_num):
            #plt.clf()
            ax.append(j)
            ay.append(y[i][j])
            if j < delimer:
                plt.plot(ax,ay,linewidth=2.0,color=colors[i])
            else:
                plt.plot(ax,ay,linewidth=2.0,linestyle=":",color=colors[i])
            plt.pause(0.001)  
            plt.ioff()
            #plt.plot(np.arange(delimer), y[i][:delimer], linewidth=2.0,color=colors[i])
            #plt.plot(np.arange(delimer, delimer + pre_num), y[i][delimer:],linewidth=2.0, linestyle=":",color=colors[i])# predict node

    # # 画结果
    # plt.figure(figsize=(35, 15))
    # plt.title("Predict values for sin function", fontsize=25)
    # plt.xlabel('x', fontsize=30)
    # plt.ylabel('y', fontsize=90)
    # colors = ['r','b','c','y','c','m','lime','silver','brown','chartreuse']
    # for i in range(input.size(0)):
    #     # print(len(np.arange(delimer)),len(y[i]))
    #     # print(len(np.arange(delimer, delimer + pre_num)),len(y[i][delimer:]))
    #     plt.plot(np.arange(delimer), y[i][:delimer], linewidth=2.0,color=colors[i])
    #     plt.plot(np.arange(delimer, delimer + pre_num), y[i][delimer:],linewidth=2.0, linestyle=":",color=colors[i])# predict node
    plt.savefig('pre_result.jpg')


if __name__ == '__main__':
    # 加载数据
    T = 20
    L = 100 #
    N = 1 # 
    plt.ion()
    count = 0
    while count <= 0:

        x = np.empty((N, L), 'float64')
        x[:] =np.array(range(L)) 
        x[:] += np.random.uniform(-4 * T, 4 * T, N).reshape(N, 1) 

        # 加入随机干扰可取消下两行注释
        # y = np.empty((N, L), 'float64')
        # y[:] = np.random.uniform(-0.1,0.1,size=(N,L)) 

        data = np.sin(x / 1.0 / T).astype('float64') #+ y[:]
        data = torch.from_numpy(data)

        # 通过设置 input_size 决定如何怎样预测：
        # 如果input_size = 3,则使用 n-2,n-1,n -> n+1
        # 如果input_size = 2,则使用 n-1,n -> n+1
        input_size = 3

        # 初始化训练用到的模型,损失函数,优化器
        sinnet = rnn.SinNet(input_size,50)
        sinnet = torch.load('predict_model.pkl', map_location=torch.device('cpu'))
        sinnet.double()

        pre_num = 50
        predict_sin(sinnet,data,pre_num,input_size)

        count+=1
        sleep(1)
    # system("pause")
