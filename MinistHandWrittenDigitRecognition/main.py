import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image
import numpy as np

# 预处理：将两个步骤整合在一起
transform = transforms.Compose({
    transforms.ToTensor(), # 转为Tensor，范围改为0-1
    transforms.Normalize((0.1307,),(0.3081)) # 数据归一化，即均值为0，标准差为1
})

# 训练数据集
train_data = MNIST(root='./data',train=True,download=False,transform=transforms.ToTensor())
train_loader = DataLoader(train_data,shuffle=True,batch_size=64)

# 测试数据集
test_data = MNIST(root='./data',train=False,download=False,transform=transforms.ToTensor())
test_loader = DataLoader(test_data,shuffle=False,batch_size=64)

# 模型
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(784,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,10) # 10个手写数字对应的10个输出

    def forward(self,x):
        x = x.view(-1,784) # 变形
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x

# CrossEntropyLoss
model = Model()
criterion = nn.CrossEntropyLoss() # 交叉熵损失，相当于Softmax+Log+NllLoss
optimizer = torch.optim.SGD(model.parameters(),0.8) # 第一个参数是初始化参数值，第二个参数是学习率

# 模型训练
def train():
    for index,data in enumerate(train_loader):
        input,target = data # input为输入数据，target为标签
        optimizer.zero_grad() # 梯度清零
        y_predict = model(input) # 模型预测
        loss = criterion(y_predict,target) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        if index % 100 == 0: # 每一百次保存一次模型，打印损失
            torch.save(model.state_dict(),"./model/model.pkl") # 保存模型
            torch.save(optimizer.state_dict(),"./model/optimizer.pkl")
            print("损失值为：%.2f" % loss.item())

# 加载模型
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load("./model/model.pkl")) # 加载保存模型的参数

# 模型测试
def test():
    correct = 0 # 正确预测的个数
    total = 0 # 总数
    with torch.no_grad(): # 测试不用计算梯度
        for data in test_loader:
            input,target = data
            output=model(input) # output输出10个预测取值，其中最大的即为预测的数
            probability,predict=torch.max(output.data,dim=1) # 返回一个元组，第一个为最大概率值，第二个为最大值的下标
            total += target.size(0) # target是形状为(batch_size,1)的矩阵，使用size(0)取出该批的大小
            correct += (predict == target).sum().item() # predict和target均为(batch_size,1)的矩阵，sum()求出相等的个数
        print("准确率为：%.2f" % (correct / total))

# 自定义手写数字识别测试
def test_mydata():
    image = Image.open('./test/test_one.png') # 读取自定义手写图片
    image = image.resize((28,28)) # 裁剪尺寸为28*28
    image = image.convert('L') # 转换为灰度图像
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.resize(1,1,28,28)
    output = model(image)
    probability,predict=torch.max(output.data,dim=1)
    print("此手写图片值为:%d,其最大概率为:%.2f" % (predict[0],probability))
    plt.title('此手写图片值为：{}'.format((int(predict))),fontname="SimHei")
    plt.imshow(image.squeeze())
    plt.show()

# 主函数
if __name__ == '__main__':
    # 自定义测试
    test_mydata()
    # 训练与测试
    # for i in range(5): # 训练和测试进行五轮
    #     train()
    #     test()