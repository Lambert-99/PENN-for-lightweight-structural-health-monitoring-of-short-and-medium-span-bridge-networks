#运行，生成一系列文件夹

import os
import numpy as np
# name = 'dynamic_NODE20_1F_v'

Freq = 200       #采样频率 
Bridgelen = 20  #桥长20m，两座桥
divnum = 100    #有限元分割

maxvelo = 20   #车辆可能的最大速度 m/s
minvelo = 10    #车辆可能的最小速度 m/s
Tmin = Bridgelen/maxvelo  #最小时间差
Tmax = Bridgelen/minvelo  #最大时间差
TminFre = int(np.floor(Tmin * Freq)) #最小时间差,乘频率向下取整
TmaxFre = int(np.ceil(Tmax * Freq)) #最大时间差,乘频率向上取整
T = TmaxFre - TminFre
gap = 10   #划分数据集的步长

t1 =  (Tmax+Tmax+Tmax+1+1)
n   =   int(t1*Freq) - TmaxFre-100
sample_num = 5000
print(n)
# t1 =  (Tmax+Tmax+Tmax+Tmax)
# n   =   int(t1*Freq) - TmaxFre
# sample_num = 6000


