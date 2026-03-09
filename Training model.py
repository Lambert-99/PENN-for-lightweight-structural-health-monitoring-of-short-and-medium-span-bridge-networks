from PECNN import NET
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
import csv
import random
import pickle
from A_parameters import  TminFre, TmaxFre, gap,n,T,Freq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import re
#是训练还是测试,训练为0，测试为1
train_state = 0
#是正常还是损伤,正常为0，损伤为1
state = 0
#损失代码
damage = "_0"
total_step   = 0
start_epoch  = 0
epochs = 2000
lr =  4e-5
# name = f"15-20+1Force+fft—filter-2hz-1axle-disp-point50+3cnn+3cnn+DATA5000"
# name = f"15-20+2Force+fft—filter-2hz-1axle-disp-point50+3cnn+3cnn+DATA5000"
name = f"15-20+2Force+fft—filter-2hz-2axle-disp-point50+3cnn+3cnn+DATA5000"
# name = f"15-20+2Force+fft—filter-2hz-2axle-diffdisp-point50+3cnn+3cnn+DATA5000"
# name = f"15-20+2Force+fft—filter-2hz-2axle-rota-point50+3cnn+3cnn+DATA5000-1"
# name = f"15-20+2Force+fft—filter-2hz-2axle-strain-point50+3cnn+3cnn+DATA5000"
log_dir  = f"./runs/II_"+name
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

val_loss1 = 1e11
def dataset_sp(dataset,train_scale=0.9,valid_scale=0.1,test_scale=0):           #划分数据集，train_dataset,valid_dataset,test_dataset
    #dataset 为三维数组，c*h*w
    data_num  = dataset.shape[0]
    train_num = int(train_scale*data_num)
    valid_num = int(valid_scale * data_num)
    test_num  = int(test_scale * data_num)
    np.random.shuffle(dataset)
    train_dataset=dataset[0:train_num,:,:]
    valid_dataset=dataset[train_num:train_num+valid_num,:,:]
    test_dataset=dataset[train_num+valid_num:train_num+valid_num+test_num,:,:]
    return train_dataset,valid_dataset,test_dataset
class mydata(Dataset):         #为输入输出数据建立索引，转为张量形式，并扩展一个维度，原始：（num，4，n），现在：（1，num，4，n）
    def __init__(self,Data_input,Data_output):
        self.Data_input = Data_input.astype(float)
        self.Data_output = Data_output.astype(float)
    def __len__(self):
        return len(self.Data_input)
    def __getitem__(self, index):
        input  = torch.FloatTensor(self.Data_input[index])
        output = torch.FloatTensor(self.Data_output[index])
        # input  = np.expand_dims(input, 0)
        # output = np.expand_dims(output, 0)
        return input, output 
if True:
    # --------------------随机种子固定
    def same_seeds(seed):
        # 为了禁止hash随机化，使得实验可复现
        os.environ['PYTHONHASHSEED'] = str(seed)
        # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        random.seed(seed)  # Python random module.
        torch.use_deterministic_algorithms(True)
        # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
        torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
        torch.backends.cudnn.deterministic = True  # 固定网络结构
    # 随机种子ccc
    seed = 1
    same_seeds(seed)
if True:
    # --------------------数据处理
    data_root = './train_data'
    os.makedirs(data_root, exist_ok=True)
    dataname  = os.path.join(data_root,name) #初始状态下训练数据位置
    os.makedirs(dataname, exist_ok=True)
    dataname00  = os.path.join(data_root,'sample_200Hz_2axle_2F_5000_train.txt') #初始状态下训练数据位置
    with open(dataname00, 'rb') as file:
        data_list = pickle.load(file)
    print(data_list.shape)   
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           #调用GPU，并确认调用的是cpu还是gpu
    print(device)
train_data,valid_data,test_data = dataset_sp(data_list)

train_input  = train_data[:,[0],0:1608]
train_input = np.transpose(train_input, (0, 2, 1))  # 将维度 0 和 1 进行转置
train_output = train_data[:,[0],1608:]
train_output = np.transpose(train_output, (0, 2, 1))  # 将维度 0 和 1 进行转置
# train_output = train_data[:,0,0:n]
valid_input  = valid_data[:,[0],0:1608]
valid_input = np.transpose(valid_input, (0, 2, 1))  # 将维度 0 和 1 进行转置
valid_output = valid_data[:,[0],1608:]
valid_output = np.transpose(valid_output, (0, 2, 1))  # 将维度 0 和 1 进行转置

    
train_num = len(train_input)
val_num   = len(valid_input)

tr_dataset=mydata(train_input,train_output)
va_dataset=mydata(valid_input,valid_output)

batch_size = 32
if train_state == 1:
    batch_size=1
train_loader = torch.utils.data.DataLoader(tr_dataset,batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
validate_loader = torch.utils.data.DataLoader(va_dataset,batch_size=1, shuffle=False,num_workers=0,pin_memory=True)
print("using {} samples for training, {} samples for validation.".format(train_num, val_num))
Net= NET(init_weights = True)
save_root  = os.path.join('./model_logs',name) 
os.makedirs(save_root, exist_ok=True)
if os.path.exists(save_root):
    # 获取目录下所有文件，并筛选出唯一的 .pth 文件
    pth_files = [f for f in os.listdir(save_root) if f.endswith(".pth")]

    if pth_files:
        pth_file = os.path.join(save_root, pth_files[0])  # 直接获取唯一的 .pth 文件
        print("发现pth文件:", pth_file)
        checkpoint = torch.load(pth_file)
        Net.load_state_dict(checkpoint['net'])
        print('加载成功！')
        # 使用正则表达式匹配科学计数法格式的数字
        match = re.search(r'[-+]?\d*\.\d+e[-+]?\d+', pth_file)
        if match:
            val_loss1 = float(match.group())
            print("提取的loss:", val_loss1)
        else:
            print("未找到匹配的数字")
    else:
        print("未发现pth文件")
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
torch.cuda.empty_cache() #释放CUDA内存
Net.to(device)
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(Net.parameters(), lr=lr)
countnum = 0
width = 50
loss_record = []
# excel_record = []
for epoch in range(start_epoch, epochs):
    string1 = f"*******************************第{epoch}次训练及验证_学习率为_{lr}"
    string2 = f"*********************************"
    string  = string1 +name+string2
    print(string)
    running_loss = 0.0
    if train_state == 0:
        Net.train()
    progress_bar = tqdm(train_loader, desc=f"训练", ncols =width, leave=False)
    for step, data in enumerate(progress_bar):
        total_step = total_step + 1
        input, output = data
        # input *= 1000
        optimizer.zero_grad()
        # output1,x11 = Net(input.to(device),output.to(device),T)
        output1 = Net(input.to(device))
        loss=loss_function(output1,output.to(device))
        if train_state == 0:
            time = np.arange(0, (1600)/Freq, 1/Freq)
            # disp1
            plt.figure()
            plt.grid(linestyle='--', )
            plt.title('Title')
            plt.xlabel('Time')
            plt.ylabel('DISP')
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.plot(time, output[0,:,0].cpu().detach().numpy(), c='b', lw=0.5, label='Real')
            plt.plot(time, output1[0,:,0].cpu().detach().numpy(), c='r', lw=0.5, label='Reconstructed')
            plt.legend(loc='upper right')
            fig_save = os.path.join('.\save',f'IR_Train_{lr}_'+name) 
            os.makedirs(fig_save, exist_ok=True)
            fig_save_reconstr = os.path.join(fig_save,f'第{epoch}次第{step + 1}轮训练对比.png')
            plt.savefig(fig_save_reconstr)
            plt.close()
        if train_state == 1:
            loss_record.append(loss.cpu().detach().numpy())
            font1 = {'family' : 'Georgia','weight' : 'normal','size'   : 4.5,}
        # time = np.arange(0, len(R_n), 1)
            fig_save = os.path.join('.\saveplot',name+damage) 
            os.makedirs(fig_save, exist_ok=True)
            #绘图，R
            figsize = 2.3,0.8
            figure, ax = plt.subplots(figsize=figsize,dpi= 3000)
            fig_save_reconstr = os.path.join(fig_save,f'第{step}_.png')
            A,=plt.plot(output[0,:,0].cpu().detach().numpy(), lw=0.4,color = 'dodgerblue',alpha = 0.5, label='Measured response')
            B,=plt.plot(output1[0,:,0].cpu().detach().numpy(), lw=0.4,color = 'r',alpha = 0.6,linestyle = (0,(5,5)), label='Reconstructed response')
            legend = plt.legend(handles=[A,B], loc='upper center', bbox_to_anchor=(0.44, 1.23), ncol=3,  frameon=False,prop=font1, handletextpad=0.7 ,columnspacing=1)

            ax.tick_params(which='major', length=1, width=0.2,labelsize=4, direction='in',left=True,right=True,bottom=True,top=True,pad=0.8) 
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            plt.xlabel('Time (s)',font1,labelpad=1)
            plt.ylabel('Displacement (mm)',font1,labelpad=1)
            plt.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17)
            bwith = 0.2
            ax.spines['bottom'].set_linewidth(bwith)
            ax.spines['left'].set_linewidth(bwith)
            ax.spines['top'].set_linewidth(bwith)
            ax.spines['right'].set_linewidth(bwith)
            
            plt.savefig(fig_save_reconstr)
            plt.close()

        if train_state == 0:
            loss.backward()
            optimizer.step()
        running_loss+=loss.item()
    Net.eval()
    train_loss = running_loss/(step+1)
    writer.add_scalar(f"Train_Loss", train_loss, epoch)
    print('【训练集】{:.4e}'.format(train_loss))
    if train_state == 0:
        
        running_loss = 0.0
        with torch.no_grad():
            # val_bar = tqdm(validate_loader)
            progress_bar = tqdm(validate_loader, desc=f"验证", ncols =width, leave=False)
            for step, val_data in enumerate(progress_bar):
                valinput, valoutput = val_data
                valoutput1 = Net(valinput.to(device))
                loss=loss_function(valoutput1,valoutput.to(device))
                running_loss+=loss.item()
                # print('迭代{}次，训练{}次，验证loss为{}'.format(epoch, step, loss))
            val_loss = running_loss/(step+1)
            print('【验证集】{:.4e}'.format(val_loss))
            writer.add_scalar("Valid_Loss", val_loss, epoch)
            
            save_path = os.path.join(save_root,'epoch {}_valloss{}.pth')
            state_dict = {"net": Net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            countnum += 1
            if val_loss<val_loss1:
                countnum = 0
                
                for filename in os.listdir(save_root):
                    file_path = os.path.join(save_root, filename) 
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)  # 删除文件
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                torch.save(state_dict, save_path.format(epoch,val_loss))
            # print('reconstructed plot over')
                val_loss1 = val_loss
        if countnum > 50:
            print("超过50epoch loss未下降")
            break

print('Finished Training')        
writer.close()