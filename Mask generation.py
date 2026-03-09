import numpy as np
import pickle
import os
import csv
from A_parameters import  TminFre, TmaxFre, gap,t1,T,Freq,n
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
def gaussian_weight(x, x0, tau):
    """
    计算高斯权重
    :param x: 自变量数组
    :param x0: 当前计算点
    :param tau: 平滑参数（带宽）
    :return: 权重数组
    """
    return np.exp(-((x - x0) ** 2) / (2 * tau ** 2))

def locally_weighted_regression(x, y, tau):
    """
    局部加权回归滤波
    :param x: 输入数据的索引（可以是时间序列）
    :param y: 输入数据序列
    :param tau: 平滑参数（带宽）
    :return: 经过滤波后的数据序列
    """
    m = len(x)
    y_pred = np.zeros(m)

    for i in range(m):
        weights = np.diag(gaussian_weight(x, x[i], tau))  # 计算当前点的权重矩阵
        X = np.vstack((np.ones(m), x)).T  # 构造设计矩阵
        theta = np.linalg.pinv(X.T @ weights @ X) @ (X.T @ weights @ y)  # 计算加权最小二乘解
        y_pred[i] = np.array([1, x[i]]) @ theta  # 计算预测值

    return y_pred    
tau = 15.0  # 平滑参数（带宽）    
if True:
    data_root = './train_data'
    os.makedirs(data_root, exist_ok=True)
    dataname  = os.path.join(data_root,'sample_200Hz_2axle_2F_5000.txt') #初始状态下桥一训练数据位置
    dataname2  = os.path.join(data_root,'sample_200Hz_2axle_2F_5000_train.txt')
#初始状态下桥一训练数据位置
    with open(dataname, 'rb') as file:
        data_list = pickle.load(file)
    print(data_list.shape)
def noise(input1,pecent):
    length = len(input1)
    #生成标准差 
    input_std   = pecent*np.std(input1)
    random_noise_input = np.random.randn(input1.shape[0])
    noisy_input = input1.copy()
    noisy_input += input_std * random_noise_input
    return noisy_input

def find_right_zero_value(sequence, index):
    scal = sequence[index]
    d = 0
    for i in range(index+1, len(sequence)):
        if sequence[i] < 5e-3*scal:
            d = i
            break
    return d 
def find_left_zero_value(sequence, index):
    scal = sequence[index]

    c = 0
    for i in range(index, -1, -1):
        if sequence[i] < 5e-3*scal:
            c = i
            break
    return c  

if True:
    #非损伤情况
    input1      = 1e3*data_list[:,0,:]            
    output1     = 1e3*data_list[:,1,:]          
    plt1 = True
    # plt1 = False
    actual_cutoff_frequency = 2  
    # 将实际截止频率归一化到Nyquist频率范围内
    normalized_cutoff_frequency = actual_cutoff_frequency / (0.5 * Freq)
    # 确定滤波器的阶数
    order = 4  # 假设阶数为4
    cout_fail  = 0
    cout_i  = 0
    # 设计巴特沃斯滤波器
    b, aa = signal.butter(order, normalized_cutoff_frequency, 'low')
    # output2     = data_list[:,2,:]            #位移通道4
    sample_list = []
    for i in range(len(input1[:,0])):
        print(i)
        cross_correlation=[]       
        if  True:
            pecent = 0.05
            input1[i,:]      = noise(input1[i,:],pecent)            #节点4、
            # input     += noise_input1
            output1[i,:]     = noise(output1[i,:],pecent)
            # output5     = noise(output5,pecent)            #节点3、5
        if plt1:

            plt.figure(figsize=(10, 6))
            plt.plot(input1[i,:], color='r',label='Time Series') 
            # plt.plot(peaks, input1[i,peaks], 'x', color='red', markersize=10, label='Peaks')
            plt.plot(output1[i,:], color='r',label='Time Series')
        time = np.arange(0, len(input1[i,:]), 1)
        tau = 10.0  # 平滑参数（带宽）
        input1[i,:] = signal.filtfilt(b, aa, input1[i,:])
        output1[i,:] = signal.filtfilt(b, aa, output1[i,:])
        aaaaa = np.max(input1[i,:])
        bbbbb = np.max(output1[i,:])
        peaks, _ = find_peaks(input1[i,:],height=aaaaa/10, distance=None, threshold=None,prominence=None,width=None,wlen=None)
        peaks1, _ = find_peaks(output1[i,:],height=bbbbb/10, distance=None, threshold=None ,prominence=None,width=None,wlen=None)

        left1 = find_left_zero_value(input1[i,:],peaks[0]) #0
        left2 = find_left_zero_value(output1[i,:],peaks1[0]) #2
        right1 = find_right_zero_value(input1[i,:],peaks[-1]) #1
        right2 = find_right_zero_value(output1[i,:],peaks1[-1]) #3       
        right3 = 2*right1-right2 #4
        left3 = 2*left2-left1 #5

        dddd = np.ones((1,8))
        dddd[0,0]=left1-50
        dddd[0,1]=left2+50
        dddd[0,2]=left2-50
        dddd[0,3]=left3+50
        dddd[0,4]=right3-50
        dddd[0,5]=right1+50
        dddd[0,6]=right1-50
        dddd[0,7]=right2+50
        if (dddd<0).any():
            print(dddd)
            plt1 = True
            plt.figure(figsize=(10, 6))
            print('不对劲')
            cout_fail  = 1
            cout_i  = i
        sample_list.append(np.concatenate((dddd,input1[[i],:],output1[[i],:] ),axis = 1))
        if plt1:
            plt.plot(input1[i,:], label='Time Series') 
            plt.plot(output1[i,:], label='Time Series') 
            plt.plot(left1, input1[i,left1], 'x', color='g', markersize=10, label='Zeros')      
            plt.plot(right1, input1[i,right1], 'x', color='g', markersize=10, label='Zeros') 
            plt.plot(left2, output1[i,left2], 'x', color='g', markersize=10, label='Zeros')      
            plt.plot(right2, output1[i,right2], 'x', color='g', markersize=10, label='Zeros')    
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Time Series with Peaks')
            plt.legend()
            plt.grid(True)
            plt.show()
print("如果是1就有问题",cout_fail)
print("i是",cout_i)
a = len(sample_list)
b = sample_list[0].shape[0]
c = sample_list[0].shape[1]
print(a,b,c)
sample_list = np.reshape(sample_list,(a,b,c))
print('sample_list SIZE :', sample_list[0].shape)
# 保存sample_list到txt文件
with open(dataname2, 'wb') as file:
    pickle.dump(sample_list, file)
print('sample_list文件已经成功保存至', dataname2)

