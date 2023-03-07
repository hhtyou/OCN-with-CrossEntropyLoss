import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import parse_args
import torchvision
import numpy as np
import random


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


class STN_Net(nn.Module):
    def __init__(self,use_stn=True):
        super(STN_Net, self).__init__()
        #self.conv1 = nn.Conv2d(3,10,kernel_size=5)  
        #self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320,50)
        #self.fc2 = nn.Linear(50,10)
        #用来判断是否使用STN
        self._use_stn = use_stn


        #localisation net
        #从输入图像中提取特征 (w-k+2p)/s+1
        #输入图片的shape为(-1,1,28,28) 3,224,224------>cat-------->3,224,224*3
        self.localization = nn.Sequential(
            #卷积输出shape为(-1,8,22,22) 8,218,218------>cat-------->8,218,666
            nn.Conv2d(9,8,kernel_size=7),
            #最大池化输出shape为(-1,8,11,11) 109,109------>cat-------->8,109,333
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            #卷积输出shape为(-1,10,7,7) 10,105,105------>cat-------->10,105,329
            nn.Conv2d(8,10,kernel_size=5),
            #最大池化层输出shape为(-1,10,3,3)10,52,52------>cat-------->10,52,164
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            #卷积输出shape为(-1,10,7,7) 48,48------>cat-------->10,48,82
            nn.Conv2d(10,10,kernel_size=5),
            #最大池化层输出shape为(-1,10,3,3)10,24,24------>cat-------->10,24,41
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            #卷积输出shape为(-1,10,7,7) 20,20------>cat-------->10,20,37
            nn.Conv2d(10,10,kernel_size=5),
            #最大池化层输出shape为(-1,10,3,3)10,10,10------>cat-------->10,10,18
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            #卷积输出shape为(-1,10,7,7) 6,6------>cat-------->10,6,14
            nn.Conv2d(10,10,kernel_size=5),
            #最大池化层输出shape为(-1,10,3,3)10,3,3------>cat-------->10,3,7
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )
        #利用全连接层回归\theta参数
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3,32),
            nn.ReLU(True),
            nn.Linear(32,3*6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0]
        ,dtype=torch.float))


    def forward(self,x):
        args = parse_args()
        if args.use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        #提取输入图像中的特征
        xs = self.localization(x)
        #xs = xs.view(-1,10*3*7)
        xs = xs.view(-1,10*3*3)
        #print("x.size(0)",x.size(0))
        #print("x.size(1)",x.size(1))
        #print("x.size(2)",x.size(2))
        #print("x.size(3)",x.size(3))
        #print("x",x.shape)
        #xs = xs.view(-1, 30 * self.out_dim * self.out_dim)
        ind1 = Variable(torch.LongTensor(range(0,2))).to(device)
        ind2 = Variable(torch.LongTensor(range(2,4))).to(device)
        ind3 = Variable(torch.LongTensor(range(4,6))).to(device)
        inp1 = Variable(torch.LongTensor(range(0,int(x.size(1)/3)))).to(device)
        inp2 = Variable(torch.LongTensor(range(int(x.size(1)/3),int(x.size(1)*2/3)))).to(device)
        inp3 = Variable(torch.LongTensor(range(int(x.size(1)*2/3),x.size(1)))).to(device)
        
        #回归theta参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1,6,3)
        #print("theta",theta.shape)
        #print(theta)
        theta_1 = torch.index_select(theta,1, ind1)#(64,2,3)
        #print("theta_1",theta_1.shape)
        #print(theta_1)
        theta_2 = torch.index_select(theta,1, ind2)
        #print("theta_2",theta_2.shape)
        theta_3 = torch.index_select(theta,1, ind3)
        #print("theta_3",theta_3.shape)
        #x(64,9,224,224)
        input_1 = torch.index_select(x, 1, inp1)#(64,3,224,224)
        #print("x",x.shape)
        #print("input_1",input_1.shape)
        input_2 = torch.index_select(x, 1, inp2)
        #print("input_2",input_2.shape)
        input_3 = torch.index_select(x, 1, inp3)


###********if you want stn and ocn Simultaneous training please comment off these and change the return********************************###
        #print("input_3",input_3.shape)
        #利用theta参数计算变换后图片的位置(根据形变参数产生sampling grid)
        grid_1 = F.affine_grid(theta_1, input_1.size(),align_corners=True)
        grid_2 = F.affine_grid(theta_2, input_2.size(),align_corners=True)
        grid_3 = F.affine_grid(theta_3, input_3.size(),align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值(对图像进行变形)
        x1 = F.grid_sample(input_1, grid_1, padding_mode="border",align_corners=True)
        x2 = F.grid_sample(input_2, grid_2, padding_mode="border",align_corners=True)
        x3 = F.grid_sample(input_3, grid_3, padding_mode="border",align_corners=True)
        #没有random时resnet的输入
        resnet_input = torch.cat((x1,x2,x3),1)
###************************************************************************************###

        #random时resnet输入
        #三张图封装进list
        #tensor_list=list()
        #tensor_list.append(x1)
        #tensor_list.append(x2)
        #tensor_list.append(x3)
        #list打乱顺序
        #random_list=random.sample(tensor_list,3)

        #random_x1=random_list[0]
        #random_x2=random_list[1]
        #random_x3=random_list[2]
        #图片串联
        #resnet_input = torch.cat((random_x1,random_x2,random_x3),1)



        #利用theta参数计算变换后图片的位置
        ##grid = F.affine_grid(theta,x.size())
        ##print(grid)
        #根据输入图片计算变换后图片位置填充的像素值
        ##x = F.grid_sample(x,grid, padding_mode="border")


        #return x
        #return theta
        #return x1,x2,x3
       # return theta_1,theta_2,theta_3,input_1,input_2,input_3,x1,x2,x3,random_x1,random_x2,random_x3,resnet_input
        return theta_1,theta_2,theta_3,input_1,input_2,input_3,x1,x2,x3,resnet_input
    #def forward(self,x):   

 #def forward(self,x):
        #使用STN模块
     #   if self._use_stn:
      #      x = self.stn(x)
        #利用STN矫正过的图片来进行图片的分类
        #经过conv1卷积输出的shape为(-1,10,24,24)
        #经过max pool的输出shape为(-1,10,12,12)
        #x = F.relu(F.max_pool2d(self.conv1(x),2))
        #经过conv2卷积输出的shape为(-1,20,8,8)
        #经过max pool的输出shape为(-1,20,4,4)
       # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
       # x = x.view(-1,320)
       # x = F.relu(self.fc1(x))
       # x = F.dropout(x,training=self.training)
       # x = self.fc2(x)

#        return F.log_softmax(x,dim=1)
      #  return x



__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=6, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 9, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        prob = F.softmax(x,dim=1) 
        return x,prob
def ResNet18():
    return ResNet([2, 2, 2, 2])
def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

def Resnet():

    net = ResNet18()

    #model = torchvision.models.resnet50()

    #print(net)

    #input = torch.randn(1, 3, 224, 224)
    #out = model(input)
    #print(out.shape)

    return net


