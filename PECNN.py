import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

###
class NET(nn.Module):
    # 创建网络，导入nn.Module模块，利用模块功能建立神经网络
    def __init__(self,init_weights=False):
        super(NET, self).__init__()
        # 超类，引入父类参数
        self.F_en1=f_encoder(kernel_size=(1,5),padding=(0,2))
        # 调用程序1：编码器： 卷积核 =   ，填充 = 
        self.F_block1=f_block2(512)
        self.F_en2=f_encoder(kernel_size=(1,5),padding=(0,2))
        self.F_en3=f_encoder(kernel_size=(1,5),padding=(0,2))
        # 调用程序1：编码器： 卷积核 =   ，填充 = 
        self.F_block2=f_block2(512)
        self.F_block3=f_block2(512)
        # 调用程序2：自定义神经网络，通道数 = 
        if init_weights:
            # 判断是否需要初始化权重
            self._initialize_weights()
    def _initialize_weights(self):
        # 函数：初始化权重
        for m in self.modules():   # 模块：对所有参数进行迭代                  
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):     # 检测对象m是否为Conv2d：卷积层/ConvTranspose2d：反卷积层模块
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')  
                # kaiming_normal初始化，使用高斯分布+激活函数
                # nn.init.xavier_uniform(m.weight)                                # xavier初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    def extract_slice(self,x, indices0, indices1):

        device = x.device
        B, C, _, D = x.shape  
        # 计算时间长度L_i 
        L = indices1 - indices0 + 1
        # 创建坐标网格
        d = torch.arange(D, device=device).view(1, 1, 1, D) 
        s = torch.arange(C, device=device).view(1, C, 1, 1)  
        # 计算时间步t 
        t = d - indices0  
        # 有效时间步掩码 
        valid_t = (t >= 0) & (t < L)
        
        # 计算区间边界
        scale = 512.0 / (L-100)  # 每个时间步对应的宽度
        s_start = ((t-100) * scale).floor()
        s_start = torch.clamp(s_start, min=0) 
        s_end = (t + 1) * scale
        s_end = torch.clamp(s_end, max=512) 
        
        # 生成掩码 
        mask = (
            (s >= s_start) &   
            (s < s_end) &     
            valid_t          
        )
    
        return  mask
    def forward(self,x):                                                          # 定义向前传播过程
        x = x.permute(0, 2, 1).unsqueeze(2)
        indices0 = x[:,:,:,[0]].int()  # (32,1,1,1)，确保索引范围合法
        indices1 = x[:,:,:,[1]].int()  # (32,1,1,1)，确保索引范围合法
        indices2 = x[:,:,:,[2]].int()  # (32,1,1,1)，确保索引范围合法
        indices3 = x[:,:,:,[3]].int()  # (32,1,1,1)，确保索引范围合法
        indices4= x[:,:,:,[4]].int()  # (32,1,1,1)，确保索引范围合法
        indices5 = x[:,:,:,[5]].int()  # (32,1,1,1)，确保索引范围合法
        indices6= x[:,:,:,[6]].int()  # (32,1,1,1)，确保索引范围合法
        indices7 = x[:,:,:,[7]].int()  # (32,1,1,1)，确保索引范围合法
        # 生成索引范围（j:j+30）
        batch_size, _, _, lent = x[:,:,:,8:].shape
            # 生成索引范围
        index_range = torch.arange(lent, device=x[:,:,:,8:].device).view(1, 1, 1, lent)  

        # 创建掩码：True 表示要保留的部分，False 表示要置为 0
        mask11 = (index_range >= indices0) & (index_range < indices1)  # 车1桥1
        mask21 = (index_range >= indices4) & (index_range < indices5)  # 车2桥1
        # 应用掩码
        x1 = x[:,:,:,8:] * mask11
        x2 = x[:,:,:,8:] * mask21
        x1=self.F_en1(x1)                                         
        x2=self.F_en2(x2)                                          

        x = torch.zeros((batch_size,512,1,lent)).to('cuda')
        x[self.extract_slice(x, indices2, indices3)] += x1[self.extract_slice(x, indices0, indices1)]
        x[self.extract_slice(x, indices6, indices7)] += x2[self.extract_slice(x, indices4, indices5)]
        F2=self.F_block3(x)
        output = F2[:,:,0,:].permute(0, 2, 1)
        return output
class BasicConv2d(nn.Module):                                                     #自定义正向卷积函数单位模块
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):     #输入的通道数、输出的通道数、卷积核、卷积步幅、卷积填充大小
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        # self.batch=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x=self.batch(x)
        return x
class BasicTConv2d(nn.Module):                                                    #自定义反向卷积函数单位模块
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):
        super(BasicTConv2d, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.5)
        # self.batch=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.tconv(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x=self.batch(x)
        return x
class f_encoder(nn.Module):                                                       #组合正向模块，形成编码器，用torch.cat进行数据融合
    def __init__(self, kernel_size,padding):
        super(f_encoder, self).__init__()
        self.conv1a=BasicConv2d(1,16,kernel_size=(1,1),stride=(1,1),padding=(0,0))      # B,16,24,200
        self.conv1b=BasicConv2d(1,16,kernel_size=(1,1),stride=(1,1),padding=(0,0))      # B,16,12,400
        self.conv1c=BasicConv2d(1,16,kernel_size=(1,1),stride=(1,1),padding=(0,0))      # B,16,1,400
        # self.conv1a_1=BasicConv2d(16,16,kernel_size=(1,1),stride=(1,1),padding=(0,4))      # B,16,1,400
        # self.conv1b_1=BasicConv2d(16,16,kernel_size=(1,1),stride=(1,1),padding=(0,4))      # B,16,1,400
        # self.conv1c_1=BasicConv2d(16,16,kernel_size=(1,1),stride=(1,1),padding=(0,4))      # B,16,1,400
        # self.conv1a = BasicConv2d(1, 16, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
        # self.conv1b = BasicConv2d(1, 16, kernel_size=(3, 9), stride=(1, 1), padding=(0, 4))
        # self.conv1c = BasicConv2d(1, 16, kernel_size=(6, 9), stride=(1, 1), padding=(0, 4))
        # self.conv1a_1 = BasicConv2d(16, 16, kernel_size=(6, 9), stride=(1, 1), padding=(0, 4))
        # self.conv1b_1 = BasicConv2d(16, 16, kernel_size=(4, 9), stride=(1, 1), padding=(0, 4))
        # self.conv1c_1 = BasicConv2d(16, 16, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
        self.encoder1=nn.Sequential(
            BasicConv2d(16, 32, kernel_size=(1,1), stride=(1, 1), padding=(0,0)),             # B,32,1,400 
            # BasicConv2d(32,32,kernel_size=kernel_size,stride=(1,1),padding=padding),
            BasicConv2d(32,64,kernel_size=kernel_size,stride=(1,1),padding=padding),          # B,64,1,400 
            BasicConv2d(64, 128, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(128, 256, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(256, 512, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            # BasicConv2d(512,512,kernel_size=kernel_size,stride=(1,1),padding=padding)         # B,512,1,400 
        )
        self.encoder2=nn.Sequential(
            BasicConv2d(16, 32, kernel_size=(1,1), stride=(1, 1), padding=(0,0)),             # B,32,1,400 
            # BasicConv2d(32,32,kernel_size=kernel_size,stride=(1,1),padding=padding),
            BasicConv2d(32,64,kernel_size=kernel_size,stride=(1,1),padding=padding),          # B,64,1,400 
            BasicConv2d(64, 128, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(128, 256, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(256, 512, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            # BasicConv2d(512,512,kernel_size=kernel_size,stride=(1,1),padding=padding)         # B,512,1,400 
        )
        self.encoder3=nn.Sequential(
            BasicConv2d(16, 32, kernel_size=(1,1), stride=(1, 1), padding=(0,0)),             # B,32,1,400 
            # BasicConv2d(32,32,kernel_size=kernel_size,stride=(1,1),padding=padding),
            BasicConv2d(32,64,kernel_size=kernel_size,stride=(1,1),padding=padding),          # B,64,1,400 
            BasicConv2d(64, 128, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(128, 256, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            BasicConv2d(256, 512, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            # BasicConv2d(512,512,kernel_size=kernel_size,stride=(1,1),padding=padding)         # B,512,1,400 
        )
        self.pool = MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
    def forward(self, x):
        x1a=self.conv1a(x)                                                 # B,16,1,400
        x1b=self.conv1b(x)                                                  # B,16,1,400
        x1c=self.conv1c(x)                                                  # B,16,1,400
        # x_r=torch.cat((x1a,x1b,x1c),1)                                                     # B,48,1,400
        # x_r=torch.cat((x1a,x1b),1)                                                     # B,48,1,400                 
        x1a=self.encoder1(x1a)
        x1b=self.encoder2(x1b)
        x1c=self.encoder3(x1c)
        x_r=torch.cat((x1a,x1b,x1c),1)                                                     # B,48,1,400
        
        x_r = self.pool(x_r[:,:,0,:])
        x_r = x_r.unsqueeze(2)
        return x_r
class f_block2(nn.Module):                                                        #组合反向模块，形成解码器
    def __init__(self,inchannel):
        super(f_block2, self).__init__()
        self.exprand1=nn.Sequential(
            BasicTConv2d(inchannel,inchannel//2,kernel_size=(3,3),stride=(1,1),padding=(1,1)),  #channel->1         200
            BasicTConv2d(inchannel//2, inchannel//4, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  #channel-> 1
            BasicTConv2d(inchannel//4, inchannel//8, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  #channel->1
            BasicTConv2d(inchannel//8, inchannel//16, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  #channel->1
            BasicTConv2d(inchannel//16, 16, kernel_size=(3,3),stride=(1,1),padding=(1,1)) #channel->1
        )
        self.exprand2=nn.Sequential(
            BasicTConv2d(inchannel,inchannel//2,kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//2, inchannel//4, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//4, inchannel//8, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//8, inchannel//16, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//16, 16, kernel_size=(3,3),stride=(1,1),padding=(1,1)) 
        )
        self.exprand3=nn.Sequential(
            BasicTConv2d(inchannel,inchannel//2,kernel_size=(3,3),stride=(1,1),padding=(1,1)), 
            BasicTConv2d(inchannel//2, inchannel//4, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//4, inchannel//8, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//8, inchannel//16, kernel_size=(3,3),stride=(1,1),padding=(1,1)),  
            BasicTConv2d(inchannel//16, 16, kernel_size=(3,3),stride=(1,1),padding=(1,1)) 
        )
        self.convend=nn.Conv2d(16,1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
    def forward(self,x):
        x1=self.exprand1(x)
        x2=self.exprand2(x)
        x3=self.exprand3(x)
        x_r=torch.cat((x1,x2,x3),1)                                                  
        
        x_r = self.pool(x_r[:,:,0,:])
        x_r = x_r.unsqueeze(2)
        x=self.convend(x_r)
        return x

# 定义一个最大池化层模块
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x

class BasicConv2d_BN(nn.Module):                                                  #归一化处理，再进行非线性变换，提取特征并增强模型的表达能力
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):
        super(BasicConv2d_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.bn=nn.BatchNorm2d(out_channels,momentum=0.1)
        self.relu=nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.5)
        # self.batch=nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x=self.bn(x)
        x = self.relu(x)
        # x=self.batch(x)
        return x
