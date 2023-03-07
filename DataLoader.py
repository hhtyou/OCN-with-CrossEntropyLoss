import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import cv2
import torch.utils.data as data
import numpy as np 
from torchvision import transforms as transforms
import random

class MyDataset(Dataset):

    def initialize(self):

    # 加载数据集
    # 如果GPU可用就用GPU,否则用CPU
        device = torch.device("cuda" if torch.cuda.is_available()
    					   else "cpu")
        #datalist = "./train.csv"
        #with open(datalist, newline='') as f:
         #   self.reader = csv.DictReader(f)#以csv形式读取
          #  self.STN = [row['STN'] for row in self.reader]

        #self.STN = sorted(self.STN)

        #self.option = [x.strip().split(' ')[0] for x in self.STN]
        #self.T_stn = [x.strip().split(' ')[1] for x in self.STN]

        datalist = "./STNtrain.csv"
        with open(datalist, newline='') as f:
            self.reader = csv.DictReader(f)#以csv形式读取
            self.STN = [row['STN'] for row in self.reader]#o1 o1T o2 o2T o3 o3T target

        self.STN = sorted(self.STN)

        self.option1 = [x.strip().split(' ')[0] for x in self.STN]

        self.T_stn1 = [x.strip().split(' ')[1] for x in self.STN]
        self.option2 = [x.strip().split(' ')[2] for x in self.STN]
        self.T_stn2 = [x.strip().split(' ')[3] for x in self.STN]
        self.option3 = [x.strip().split(' ')[4] for x in self.STN]
        self.T_stn3 = [x.strip().split(' ')[5] for x in self.STN]
        self.T_bento_path = [x.strip().split(' ')[6] for x in self.STN]
        self.label = [x.strip().split(' ')[7] for x in self.STN]


        #self.label = list(self.label)
        #self.label = np.array(self.label,dtype = 'float64')
        #self.label = torch.from_numpy(self.label)
        #self.label = torch.tensor(self.label)
        
    def __getitem__(self,index):
        option1 = self.option1[index]
        T_stn1 = self.T_stn1[index]
        option2 = self.option2[index]
        T_stn2 = self.T_stn2[index]
        option3 = self.option3[index]
        T_stn3 = self.T_stn3[index]
        T_bento_path = self.T_bento_path[index]
        label = self.label[index]
        label = np.array(label).astype(int)
        label = torch.from_numpy(label)
        #print(label)


        image_to_tensor = ToTensor()
        T_stn1 = Image.open(T_stn1).convert('RGB')
        T_stn1 = T_stn1.resize((224, 224), Image.BICUBIC)#cnn256
        T_stn2 = Image.open(T_stn2).convert('RGB')
        T_stn2 = T_stn2.resize((224, 224), Image.BICUBIC)#cnn256
        T_stn3 = Image.open(T_stn3).convert('RGB')
        T_stn3 = T_stn3.resize((224, 224), Image.BICUBIC)#cnn256
        T_bento = Image.open(T_bento_path).convert('RGB')
        T_bento = T_bento.resize((224, 224), Image.BICUBIC)#cnn256
        T_bento_tensor = image_to_tensor(T_bento)





        #随机翻转旋转图片
        #transform = transforms.RandomChoice([transforms.RandomApply(45),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()],p=0.5)
        #transform = transforms.RandomApply(
         #   [transforms.RandomRotation(45),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()],p=0.4
        #)
        #transform = transforms.RandomApply(
            #[transforms.RandomRotation(45),transforms.RandomHorizontalFlip()],p=0.4
  #      )
        transform = transforms.RandomApply(
            [transforms.RandomRotation(45),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()],p=0.4
        )
        #T_stn1 = transform(T_stn1)
        #T_stn1 = T_stn1.resize((224, 224), Image.BICUBIC)#cnn256
        #T_stn2 = transform(T_stn2)
        #T_stn3 = transform(T_stn3)
        #T_stn3 = transform(T_stn3)

        T_stn1_tensor = image_to_tensor(T_stn1)
        T_stn2_tensor = image_to_tensor(T_stn2)
        T_stn3_tensor = image_to_tensor(T_stn3)

        option1 = Image.open(option1).convert('RGB')
        option1 = option1.resize((224, 224), Image.BICUBIC)
        option2 = Image.open(option2).convert('RGB')
        option2 = option2.resize((224, 224), Image.BICUBIC)
        option3 = Image.open(option3).convert('RGB')
        option3 = option3.resize((224, 224), Image.BICUBIC)

        option1 = transform(option1)
        option2 = transform(option2)
        option3 = transform(option3)

        option1_tensor = image_to_tensor(option1)
        option2_tensor = image_to_tensor(option2)
        option3_tensor = image_to_tensor(option3)
        option_tensor = torch.cat((option1_tensor,option2_tensor,option3_tensor),0)
        #net_input = torch.cat((T_stn1_tensor,T_stn2_tensor,T_stn3_tensor),0) ###if you want stn and ocn Simultaneous training


        #随机图片
        tensor_list = list()
        tensor_list.append(T_stn1_tensor)
        tensor_list.append(T_stn2_tensor)
        tensor_list.append(T_stn3_tensor)

        random_list = random.sample(tensor_list,3)
        r1 = random_list[0]
        r2 = random_list[1]
        r3 = random_list[2]
        res_input = torch.cat((r1,r2,r3),0)
        r_label = [0,0,0,0,0,0]
        r_num = 0
        #按随机图片输出label
        if label.equal(torch.tensor(0)):
            if r1.equal(T_stn1_tensor) and r2.equal(T_stn2_tensor) and  r3.equal(T_stn3_tensor):
                r_label = [1,0,0,0,0,0]
                r_num = 0
            elif r1.equal(T_stn1_tensor) and r2.equal(T_stn3_tensor) and  r3.equal(T_stn2_tensor):
                r_label = [0,1,0,0,0,0]
                r_num = 1
            elif r1.equal(T_stn2_tensor) and r2.equal(T_stn1_tensor) and  r3.equal(T_stn3_tensor):
                r_label = [0,0,1,0,0,0]
                r_num = 2
            elif r1.equal(T_stn2_tensor) and r2.equal(T_stn3_tensor) and  r3.equal(T_stn1_tensor):
                r_label = [0,0,0,0,1,0]
                r_num = 4
            elif r1.equal(T_stn3_tensor) and r2.equal(T_stn1_tensor) and  r3.equal(T_stn2_tensor):
                r_label = [0,0,0,1,0,0]
                r_num = 3
            elif r1.equal(T_stn3_tensor) and r2.equal(T_stn2_tensor) and  r3.equal(T_stn1_tensor):
                r_label = [0,0,0,0,0,1]
                r_num = 5

        elif label.equal(torch.tensor(1)):
        #elif label==1:
            if r1.equal(T_stn1_tensor) and r2.equal(T_stn2_tensor) and  r3.equal(T_stn3_tensor):
                r_label = [0,1,0,0,0,0]
                r_num = 1
            elif r1.equal(T_stn1_tensor) and r2.equal(T_stn3_tensor) and  r3.equal(T_stn2_tensor):
                r_label = [1,0,0,0,0,0]
                r_num = 0
            elif r1.equal(T_stn2_tensor) and r2.equal(T_stn1_tensor) and  r3.equal(T_stn3_tensor):
                r_label = [0,0,0,1,0,0]
                r_num = 3
            elif r1.equal(T_stn2_tensor) and r2.equal(T_stn3_tensor) and  r3.equal(T_stn1_tensor):
                r_label = [0,0,0,0,0,1]
                r_num = 5
            elif r1.equal(T_stn3_tensor) and r2.equal(T_stn1_tensor) and  r3.equal(T_stn2_tensor):
                r_label = [0,0,1,0,0,0]
                r_num = 2
            elif r1.equal(T_stn3_tensor) and r2.equal(T_stn2_tensor) and  r3.equal(T_stn1_tensor):
                r_label = [0,0,0,0,1,0]
                r_num = 4
        random_label_tensor=torch.tensor(r_label) 
        r_num = np.array(r_num).astype(int)
        r_num = torch.from_numpy(r_num)
        #print(random_label_tensor)
        #print(random_label_tensor.type)


        return res_input,T_stn1_tensor,T_stn2_tensor,T_stn3_tensor,r1,r2,r3,T_bento_tensor,r_num,random_label_tensor
       # return option_tensor
    def __len__(self):
        return len(self.option1)



def get_dataloader(batch_size):
    # 加载训练集
    dataset = MyDataset()
    dataset.initialize()
    train_dataloader = data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    #test_dataloader = data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    #train(net,args.epoch_nums,args.lr,train_loader,args.batch_size,device)
    #train(train_loader,model,criterion,optimizer,epoch,device)

    #train_dataloader = torch.utils.data.DataLoader(
     #   datasets.MNIST(root="dataset", train=True, download=True,
      #                 transform=transforms.Compose([
       #                transforms.ToTensor(),
        #               transforms.Normalize((0.1307,), (0.3081,))
         #              ])), batch_size=batch_size, shuffle=True)

    # 加载测试集
    #test_dataloader = torch.utils.data.DataLoader(
     #   datasets.MNIST(root="dataset", train=False,
      #                 transform=transforms.Compose([
       #                transforms.ToTensor(),
        #               transforms.Normalize((0.1307,), (0.3081,))
         #              ])), batch_size=batch_size, shuffle=True)

    #return train_dataloader,test_dataloader
    return train_dataloader

def tensor_to_array(img_tensor):
    img_array = img_tensor.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    img_array = std * img_array + mean
    img = np.clip(img_array,0,1)
    return img
