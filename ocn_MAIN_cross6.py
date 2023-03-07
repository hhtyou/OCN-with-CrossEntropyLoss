from Net import STN_Net,Resnet
#from Visualize import visualize_stn
#from train import train
from config import parse_args
from DataLoader import get_dataloader, tensor_to_array
from TestLoader import get_testloader
#from evaluate import evaluate
import os
import torch,torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from utils import AverageMeter
import time
from torchvision.transforms import ToTensor,ToPILImage
import numpy as np
import cv2
import random


stn_best_loss =1
order_best_loss =1
best_total_loss =1
best_total_epoch = 0
stn_best_epoch=0
order_best_epoch=0
start_epoch =-1
a=0
a=np.array(a)
a=torch.from_numpy(a)

b=1
b=np.array(b)
b=torch.from_numpy(b)
def main():
#if __name__ == "__main__":
    global random_label, stn_best_loss,order_best_loss,stn_best_epoch,order_best_epoch,best_total_loss,best_total_epoch,start_epoch

    args = parse_args()
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    #加载数据集
    #train_loader,test_loader = get_dataloader(args.batch_size)
    train_loader = get_dataloader(args.batch_size)
    test_loader = get_testloader(args.val_batch_size)
    #test_loader = get_dataloader(args.val_batch_size)
    #创建网络
    #net = STN_Net(args.use_stn).to(device)
    #net = StnResNet().to(device)
    #net = AllNet().to(device)
    net2 = Resnet().to(device)
    #if args.RESUME == True:
    #    path_checkpoint1 = "pretrain/net1_epoch_4074.pth"   #4006 4300
    #    path_checkpoint2 = "pretrain/net2_epoch_759.pth"

    #    checkpoint1 = torch.load(path_checkpoint1)
    #    checkpoint2 = torch.load(path_checkpoint2)

    #    net1.load_state_dict(checkpoint1['net1'])
    #    net2.load_state_dict(checkpoint2['net2'])

        #optimizer1.load_state_dict(checkpoint1['optimizer1'])
        #optimizer2.load_state_dict(checkpoint2['optimizer'])

        #start_epoch = checkpoint2['epoch']
        #for param in net1.parameters():
         #   param.requires_grad = False

    #optimizer1 = optim.SGD(net1.parameters(),lr=args.lr)
    #optimizer2 = optim.SGD(net2.parameters(),lr=args.lr)
    #optimizer1 = optim.Adam(net1.parameters(),lr=args.lr)
    optimizer2 = optim.Adam(net2.parameters(),lr=args.lr)

    #net1.load_state_dict(torch.load('checkpoint/1963.pth'),strict = False)

    #训练模型

    #writer = SummaryWriter('./logs',comment='_scalars', filename_suffix="12345678")
    for epoch in range(start_epoch+1,args.epoch_nums):
        since=time.time()
        #loss,train_input_1,train_input_2,train_input_3,train_x1,train_x2,train_x3,train_T1,train_T2,train_T3=train(net,model,args.lr,train_loader,args.batch_size,device)
        train_acc,order_losses,train_T1,train_T2,train_T3,train_I1,train_I2,train_I3,train_I4,train_I5,train_I6,train_target,train_prob,train_predicted_label,train_label=train(net2,optimizer2,args.lr,train_loader,args.batch_size,device)
        eval_acc,eval_order_losses,eval_T1,eval_T2,eval_T3,eval_I1,eval_I2,eval_I3,eval_I4,eval_I5,eval_I6,eval_target,eval_prob,eval_predicted_label,eval_label=evaluate(net2,test_loader,args.val_batch_size,device)
          
########eval_loss,val_input_1,val_input_2,val_input_3,val_x1,val_x2,val_x3,val_T1,val_T2,val_T3=evaluate(net,test_loader,args.batch_size,device)
        #loss=train(net,args.lr,train_loader,args.batch_size,device)
        #eval_loss=evaluate(net,test_loader,args.batch_size,device)


        tensor_to_image = ToPILImage()

        #train_input_1 = train_input_1[0,...]
        #train_input_1 =tensor_to_image(train_input_1)
        #train_input_2 = train_input_2[0,...]
        #train_input_2 =tensor_to_image(train_input_2)
        #train_input_3 = train_input_3[0,...]
        #train_input_3 =tensor_to_image(train_input_3)


        #train_random_x1 = train_random_x1[0,...]
        #train_random_x1 =tensor_to_image(train_random_x1)
        #train_random_x2 = train_random_x2[0,...]
        #train_random_x2 =tensor_to_image(train_random_x2)
        #train_random_x3 = train_random_x3[0,...]
        #train_random_x3 =tensor_to_image(train_random_x3)

        #train_x1_mask = train_mask_rice_tensor[0,...]
        #train_x1_mask =tensor_to_image(train_x1_mask)
        #train_x2_mask = train_mask_op1_tensor[0,...]
        #train_x2_mask =tensor_to_image(train_x2_mask)
        #train_x3_mask = train_mask_op2_tensor[0,...]
        #train_x3_mask =tensor_to_image(train_x3_mask)

        train_T1 = train_T1[0,...]
        train_T1 =tensor_to_image(train_T1)
        train_T2 = train_T2[0,...]
        train_T2 =tensor_to_image(train_T2)
        train_T3 = train_T3[0,...]
        train_T3 =tensor_to_image(train_T3)

        #eval_input_1 = eval_input_1[0,...]
        #eval_input_1 =tensor_to_image(eval_input_1)
        #eval_input_2 = eval_input_2[0,...]
        #eval_input_2 =tensor_to_image(eval_input_2)
        #eval_input_3 = eval_input_3[0,...]
        #eval_input_3 =tensor_to_image(eval_input_3)



        #eval_random_x1 = eval_random_x1[0,...]
        #eval_random_x1 =tensor_to_image(eval_random_x1)
        #eval_random_x2 = eval_random_x2[0,...]
        #eval_random_x2 =tensor_to_image(eval_random_x2)
        #eval_random_x3 = eval_random_x3[0,...]
        #eval_random_x3 =tensor_to_image(eval_random_x3)

        #eval_x1_mask = eval_mask_rice_tensor[0,...]
        #eval_x1_mask =tensor_to_image(eval_x1_mask)
        #eval_x2_mask = eval_mask_op1_tensor[0,...]
        #eval_x2_mask =tensor_to_image(eval_x2_mask)
        #eval_x3_mask = eval_mask_op2_tensor[0,...]
        #eval_x3_mask =tensor_to_image(eval_x3_mask)

        eval_T1 = eval_T1[0,...]
        eval_T1 =tensor_to_image(eval_T1)
        eval_T2 = eval_T2[0,...]
        eval_T2 =tensor_to_image(eval_T2)
        eval_T3 = eval_T3[0,...]
        eval_T3 =tensor_to_image(eval_T3)


        train_I1 = train_I1[0,...]
        train_I1 = tensor_to_image(train_I1)
        train_I2 = train_I2[0,...]
        train_I2 = tensor_to_image(train_I2)
        train_I3 = train_I3[0,...]
        train_I3 = tensor_to_image(train_I3)
        train_I4 = train_I4[0,...]
        train_I4 = tensor_to_image(train_I4)
        train_I5 = train_I5[0,...]
        train_I5 = tensor_to_image(train_I5)
        train_I6 = train_I6[0,...]
        train_I6 = tensor_to_image(train_I6)
        train_target = train_target[0,...]
        train_target = tensor_to_image(train_target)

        #train_total_image = train_total_image[0,...]
        #train_total_image = tensor_to_image(train_total_image)

        eval_I1 = eval_I1[0,...]
        eval_I1 = tensor_to_image(eval_I1)
        eval_I2 = eval_I2[0,...]
        eval_I2 = tensor_to_image(eval_I2)
        eval_I3 = eval_I3[0,...]
        eval_I3 = tensor_to_image(eval_I3)
        eval_I4 = eval_I4[0,...]
        eval_I4 = tensor_to_image(eval_I4)
        eval_I5 = eval_I5[0,...]
        eval_I5 = tensor_to_image(eval_I5)
        eval_I6 = eval_I6[0,...]
        eval_I6 = tensor_to_image(eval_I6)
        eval_target = eval_target[0,...]
        eval_target = tensor_to_image(eval_target)

        #eval_total_image = eval_total_image[0,...]
        #eval_total_image = tensor_to_image(eval_total_image)

        if not os.path.isdir('./train_img'):
            os.makedirs('./train_img')
        if not os.path.isdir('./val_img'):
            os.makedirs('./val_img')

        #if not os.path.isdir('./logs'):
         #   os.makedirs('./logs')

        if not os.path.isdir('./log'):
            os.makedirs('./log')

        if not os.path.isdir('./checkpoint'):
            os.makedirs('./checkpoint')
        if epoch%1==0:
            #save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
             #   checkpoint =  {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),'train_loss':loss}
              #  state = net.state_dict()
               # torch.save(state,os.path.join('checkpoint','epoch_{}.pth'.format(epoch)))
        #if batch_idx % batchsize == 0:
                #print(batch_idx)
                #print(len(data))
                #print(len(train_dataloader.dataset))

            
            #train_input_1.save('train_img/epoch_{}_1_input.png'.format(epoch))
            #train_input_2.save('train_img/epoch_{}_2_input.png'.format(epoch))
            #train_input_3.save('train_img/epoch_{}_3_input.png'.format(epoch))


            #train_random_x1.save('train_img/epoch_{}_random_1_x.png'.format(epoch))
            #train_random_x2.save('train_img/epoch_{}_random_2_x.png'.format(epoch))
            #train_random_x3.save('train_img/epoch_{}_random_3_x.png'.format(epoch))

            #train_x1_mask.save('train_img/epoch_{}_1_x_mask.png'.format(epoch))
            #train_x2_mask.save('train_img/epoch_{}_2_x_mask.png'.format(epoch))
            #train_x3_mask.save('train_img/epoch_{}_3_x_mask.png'.format(epoch))

            train_T1.save('train_img/epoch_{}_1_T.png'.format(epoch))
            train_T2.save('train_img/epoch_{}_2_T.png'.format(epoch))
            train_T3.save('train_img/epoch_{}_3_T.png'.format(epoch))

            train_I1.save('train_img/epoch_{}_order_I1.png'.format(epoch))
            train_I2.save('train_img/epoch_{}_order_I2.png'.format(epoch))
            train_I3.save('train_img/epoch_{}_order_I3.png'.format(epoch))
            train_I4.save('train_img/epoch_{}_order_I4.png'.format(epoch))
            train_I5.save('train_img/epoch_{}_order_I5.png'.format(epoch))
            train_I6.save('train_img/epoch_{}_order_I6.png'.format(epoch))
            train_target.save('train_img/epoch_{}_order_target.png'.format(epoch))
            #train_total_image.save('train_img/epoch_{}_train_total_image.png'.format(epoch))
            
            #eval_input_1.save('val_img/epoch_{}_1_input.png'.format(epoch))
            #eval_input_2.save('val_img/epoch_{}_2_input.png'.format(epoch))
            #eval_input_3.save('val_img/epoch_{}_3_input.png'.format(epoch))


            #eval_random_x1.save('val_img/epoch_{}_random_1_x.png'.format(epoch))
            #eval_random_x2.save('val_img/epoch_{}_random_2_x.png'.format(epoch))
            #eval_random_x3.save('val_img/epoch_{}_random_3_x.png'.format(epoch))

            #eval_x1_mask.save('val_img/epoch_{}_1_x_mask.png'.format(epoch))
            #eval_x2_mask.save('val_img/epoch_{}_2_x_mask.png'.format(epoch))
            #eval_x3_mask.save('val_img/epoch_{}_3_x_mask.png'.format(epoch))

            eval_T1.save('val_img/epoch_{}_1_T.png'.format(epoch))
            eval_T2.save('val_img/epoch_{}_2_T.png'.format(epoch))
            eval_T3.save('val_img/epoch_{}_3_T.png'.format(epoch))

            eval_I1.save('val_img/epoch_{}_order_I1.png'.format(epoch))
            eval_I2.save('val_img/epoch_{}_order_I2.png'.format(epoch))
            eval_I3.save('val_img/epoch_{}_order_I3.png'.format(epoch))
            eval_I4.save('val_img/epoch_{}_order_I4.png'.format(epoch))
            eval_I5.save('val_img/epoch_{}_order_I5.png'.format(epoch))
            eval_I6.save('val_img/epoch_{}_order_I6.png'.format(epoch))
            eval_target.save('val_img/epoch_{}_order_target.png'.format(epoch))
            #eval_total_image.save('val_img/epoch_{}_eval_total_image.png'.format(epoch))


            with open(os.path.join('train_img','epoch_{}_prob.txt'.format(epoch)),"w") as f:
                f.write(str(train_prob))

            with open(os.path.join('val_img','epoch_{}_prob.txt'.format(epoch)),"w") as f:
                f.write(str(eval_prob))

            with open(os.path.join('train_img','epoch_{}_predlabel.txt'.format(epoch)),"w") as f:
                f.write(str(train_predicted_label))

            with open(os.path.join('val_img','epoch_{}_predlabel.txt'.format(epoch)),"w") as f:
                f.write(str(eval_predicted_label))

            with open(os.path.join('train_img','epoch_{}_labelprob.txt'.format(epoch)),"w") as f:
                f.write(str(train_label))

            with open(os.path.join('val_img','epoch_{}_labelprob.txt'.format(epoch)),"w") as f:
                f.write(str(eval_label))

            #stn_is_best = stn_losses<=stn_best_loss
            #order_is_best = order_losses<=order_best_loss
            #stn_best_loss = min(stn_losses,stn_best_loss)
            #order_best_loss = min(order_losses,order_best_loss)
            #if stn_losses<stn_best_loss:
            #    stn_best_epoch = epoch
             #   stn_best_loss = stn_best_loss




            #state1 = net1.state_dict()
            #state2 = net2.state_dict()
            if order_losses<best_total_loss:
                best_total_epoch = epoch
                best_total_loss = order_losses

                checkpoint2 =  {'epoch': epoch,'net2': net2.state_dict(), 'optimizer': optimizer2.state_dict()}
                torch.save(checkpoint2,os.path.join('checkpoint','net2_epoch_{}.pth'.format(epoch)))

           # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            #print('Train Epoch: {} train_Loss: {:.6f} val_Loss: {:.6f}'.format(
            #    epoch, loss, eval_loss))
            #print('min_loss:',best_loss,'best_epoch',best_epoch)


            print('Train Epoch: {}'.format(epoch))
            print('order_losses: {:.6f}'.format(order_losses))
            print('train_acc: {:.6f}'.format(train_acc))
            print('val_order_losses: {:.6f}'.format(eval_order_losses))
            print('val_acc: {:.6f}'.format(eval_acc))
            #print('train_total: {:.6f}'.format(train_total))
            print('best_total_loss: {:.6f}'.format(best_total_loss),'best_total_epoch',best_total_epoch)
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        if epoch%1==0:
            writer = SummaryWriter('./log')
            #log_dir = os.path.join('./logs','train')
            #train_writer = SummaryWriter(log_dir=log_dir)
            #log_dir = os.path.join('./logs','test')
            #test_writer = SummaryWriter(log_dir=log_dir)

            #train_writer.add_scalar("stn_losses",stn_losses,epoch)
            #test_writer.add_scalar("stn_losses",eval_stn_losses,epoch)
            #train_writer.add_scalar("order_losses",order_losses,epoch)
            #test_writer.add_scalar("order_losses",eval_order_losses,epoch)

            #writer.add_scalar("total_train_losses",train_total,epoch)
            #writer.add_scalar("total_val_losses",eval_total,epoch)
            writer.add_scalar("order_train_losses",order_losses,epoch)
            writer.add_scalar("order_val_losses",eval_order_losses,epoch)

            writer.add_scalar("order_train_acc",train_acc,epoch)
            writer.add_scalar("order_val_acc",eval_acc,epoch)

            #writer.add_scalar("stn_acc",stn_acc,epoch)
            #writer.add_scalar("stn_acc",eval_stn_acc,epoch)
            #writer.add_scalar("order_acc",order_acc,epoch)
            #writer.add_scalar("order_acc",eval_order_acc,epoch)
            ########writer.add_scalar("val_loss",eval_loss,epoch)         
            writer.close()
    #if args.use_eval:
        #评估模型
     #   evaluate(net,test_loader,device)
    #if args.use_visual:
        #可视化展示效果
     #   visualize_stn(net,test_loader,device)




def train(net2,optimizer2,lr,train_dataloader,batchsize,device):
    losses = 0
    train_correct_sample=0
    stn_losses = 0
    random_label =0

    #使用训练模式
    #net1.train()
    net2.train()

    #order_losses = AverageMeter()



    #选择梯度下降优化算法
    #optimizer1 = optim.SGD(net1.parameters(),lr=lr)
    #optimizer2 = optim.SGD(net2.parameters(),lr=lr)
    #optimizer1 = optim.Adam(net1.parameters(),lr=args.lr)
    #optimizer2 = optim.Adam(net2.parameters(),lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    #训练模型
    #for epoch in range(epoch_nums):
        #for batch_idx,(data,label) in enumerate(train_dataloader):
    for i,data in enumerate(train_dataloader):
        res_input,input1,input2,input3,T1,T2,T3,targets,label,random_label_tensor = data #ocn_input input1 input2 input3 randominput randominput randominput groundtruth_image groundtruth_prob

        res_input,targets,label,random_label_tensor= res_input.to(device),targets.to(device),label.to(device,dtype=torch.long),random_label_tensor.to(device,dtype=torch.float)
        input1,input2,input3,T1,T2,T3 = input1.to(device), input2.to(device), input3.to(device), T1.to(device), T2.to(device), T3.to(device)
        optimizer2.zero_grad()



###********if you want stn and ocn Simultaneous training,use these********************************###

        #theta_1,theta_2,theta_3,input_1,input_2,input_3 = net(net_input)  #输出是theta
        #theta_1,theta_2,theta_3,input_1,input_2,input_3,x1,x2,x3,res_input = net1(net_input)

　　　　#if input1 = T1:
            #grid_1 = F.affine_grid(theta_1, T1.size(),align_corners=True)
            #x1 = F.grid_sample(T1, grid_1, padding_mode="border",align_corners=True)

            #if input2 = T2:
                #grid_1 = F.affine_grid(theta_2, T2.size(),align_corners=True)
                #x1 = F.grid_sample(T2, grid_1, padding_mode="border",align_corners=True)
            #elif input2 = T3:
                #grid_2 = F.affine_grid(theta_2, T3.size(),align_corners=True)
                #x2 = F.grid_sample(T3, grid_2, padding_mode="border",align_corners=True)
                #if input3 = T3:
                    #grid_3 = F.affine_grid(theta_3, T3.size(),align_corners=True)
                    #x3 = F.grid_sample(T3, grid_3, padding_mode="border",align_corners=True)
                #elif input3 = T2:
                    #grid_3 = F.affine_grid(theta_3, T2.size(),align_corners=True)
                    #x3 = F.grid_sample(T2, grid_3, padding_mode="border",align_corners=True)
　　　　#elif input1 = T2:
            #grid_1 = F.affine_grid(theta_1, T2.size(),align_corners=True)
            #x1 = F.grid_sample(T2, grid_1, padding_mode="border",align_corners=True)

            #if input2 = T1:
                #grid_1 = F.affine_grid(theta_2, T1.size(),align_corners=True)
                #x1 = F.grid_sample(T1, grid_1, padding_mode="border",align_corners=True)
            #elif input2 = T3:
                #grid_2 = F.affine_grid(theta_2, T3.size(),align_corners=True)
                #x2 = F.grid_sample(T3, grid_2, padding_mode="border",align_corners=True)
                #if input3 = T3:
                    #grid_3 = F.affine_grid(theta_3, T3.size(),align_corners=True)
                    #x3 = F.grid_sample(T3, grid_3, padding_mode="border",align_corners=True)
                #elif input3 = T1:
                    #grid_3 = F.affine_grid(theta_3, T1.size(),align_corners=True)
                    #x3 = F.grid_sample(T1, grid_3, padding_mode="border",align_corners=True)


　　　　#elif input1 = T3:
            #grid_1 = F.affine_grid(theta_1, T3.size(),align_corners=True)
            #x1 = F.grid_sample(T3, grid_1, padding_mode="border",align_corners=True)

            #if input2 = T1:
                #grid_1 = F.affine_grid(theta_2, T1.size(),align_corners=True)
                #x1 = F.grid_sample(T1, grid_1, padding_mode="border",align_corners=True)
            #elif input2 = T2:
                #grid_2 = F.affine_grid(theta_2, T2.size(),align_corners=True)
                #x2 = F.grid_sample(T2, grid_2, padding_mode="border",align_corners=True)
                #if input3 = T2:
                    #grid_3 = F.affine_grid(theta_3, T2.size(),align_corners=True)
                    #x3 = F.grid_sample(T2, grid_3, padding_mode="border",align_corners=True)
                #elif input3 = T1:
                    #grid_3 = F.affine_grid(theta_3, T1.size(),align_corners=True)
                    #x3 = F.grid_sample(T1, grid_3, padding_mode="border",align_corners=True)

                #grid_3 = F.affine_grid(T3, input_2.size(),align_corners=True)

                #x3 = F.grid_sample(T3, grid_3, padding_mode="border",align_corners=True)
            
        #res_input = torch.cat((x1,x2,x3),1)
###*********************************************************************************************************###
        output,prob = net2(res_input) # pred_label pred_prob


#####mask image of STN's output#####
        mask_rice_tensor = (T1 >0).float()
        mask_op1_tensor = (T2 >0).float()
        mask_op2_tensor = (T3 >0).float()

#####get six image about composition#####
#rice op1 op2
        img1_hollow1 = torch.mul(T1,mask_op1_tensor) #第一层挖空
        img1_bg1_hollow1 = T1-img1_hollow1 #第一层+挖空
        img1_bg2 = img1_bg1_hollow1 + T2#第二层
        img1_hollow2 = torch.mul(img1_bg2,mask_op2_tensor)#第二层挖空
        img1_bg2_hollow2 = img1_bg2 - img1_hollow2#第二层+挖空
        I1 = img1_bg2_hollow2 + T3
#rice op2 op1
        img2_hollow1 = torch.mul(T1,mask_op2_tensor) #第一层挖空
        img2_bg1_hollow1 = T1-img2_hollow1 #第一层+挖空
        img2_bg2 = img2_bg1_hollow1 + T3#第二层
        img2_hollow2 = torch.mul(img2_bg2,mask_op1_tensor)#第二层挖空
        img2_bg2_hollow2 = img2_bg2 - img2_hollow2#第二层+挖空
        I2 = img2_bg2_hollow2 + T2

#op1 rice op2
        img3_hollow1 = torch.mul(T2,mask_rice_tensor) #第一层挖空
        img3_bg1_hollow1 = T2-img3_hollow1 #第一层+挖空
        img3_bg2 = img3_bg1_hollow1 + T1#第二层
        img3_hollow2 = torch.mul(img3_bg2,mask_op2_tensor)#第二层挖空
        img3_bg2_hollow2 = img3_bg2 - img3_hollow2#第二层+挖空
        I3 = img3_bg2_hollow2 + T3

#op1 op2 rice
        img4_hollow1 = torch.mul(T2,mask_op2_tensor) #第一层挖空
        img4_bg1_hollow1 = T2-img4_hollow1 #第一层+挖空
        img4_bg2 = img4_bg1_hollow1 + T3#第二层
        img4_hollow2 = torch.mul(img4_bg2,mask_rice_tensor)#第二层挖空
        img4_bg2_hollow2 = img4_bg2 - img4_hollow2#第二层+挖空
        I4 = img4_bg2_hollow2 + T1
#op2 rice op1
        img5_hollow1 = torch.mul(T3,mask_rice_tensor) #第一层挖空
        img5_bg1_hollow1 = T3-img5_hollow1 #第一层+挖空
        img5_bg2 = img5_bg1_hollow1 + T1#第二层
        img5_hollow2 = torch.mul(img5_bg2,mask_op1_tensor)#第二层挖空
        img5_bg2_hollow2 = img5_bg2 - img5_hollow2#第二层+挖空
        I5 = img5_bg2_hollow2 + T2
#op2 op1 rice
        img6_hollow1 = torch.mul(T3,mask_op1_tensor) #第一层挖空
        img6_bg1_hollow1 = T3-img6_hollow1 #第一层+挖空
        img6_bg2 = img6_bg1_hollow1 + T2#第二层
        img6_hollow2 = torch.mul(img6_bg2,mask_rice_tensor)#第二层挖空
        img6_bg2_hollow2 = img6_bg2 - img6_hollow2#第二层+挖空
        I6 = img6_bg2_hollow2 + T1


#####STN loss#####
        #stn_loss1 = criterion_stn(x1,T_stn1_tensor)
        #stn_loss2 = criterion_stn(x2,T_stn2_tensor)
        #stn_loss3 = criterion_stn(x3,T_stn3_tensor)
        #stn_loss = stn_loss1 + stn_loss2 + stn_loss3
        #train_acc.update(acc.item(),inputs.size(0))

#####get P1~p6 from order network##### 
       # index1=torch.LongTensor([[0]]).to(device)
       # output1=torch.gather(output,1,index1)

#        index2=torch.LongTensor([[1]]).to(device)
#        output2=torch.gather(output,1,index2)

#        index3=torch.LongTensor([[2]]).to(device)
#        output3=torch.gather(output,1,index3)

#        index4=torch.LongTensor([[3]]).to(device)
#        output4=torch.gather(output,1,index4)
#        index5=torch.LongTensor([[4]]).to(device)
#        output5=torch.gather(output,1,index5)
#        index6=torch.LongTensor([[5]]).to(device)
#        output6=torch.gather(output,1,index6)

######order loss#####
#        loss1 = criterion(I1, targets)
#        loss2 = criterion(I2, targets)
#        loss3 = criterion(I3, targets)
#        loss4 = criterion(I4, targets)
#        loss5 = criterion(I5, targets)
#        loss6 = criterion(I6, targets)
        #print("#########",loss6)
        #print("$$$$$$$$$",output)
#        loss = output1*loss1+output2*loss2+output3*loss3+output4*loss4+output5*loss5+output6*loss6
       

        #loss = criterion(output,label)
        #print(prob.type)
        #print(random_label_tensor.type)
        loss = criterion(prob,random_label_tensor) #pred_prob groundtruth_prob
        _, predicted_label = torch.max(output, 1) #pred_label
        #loss = criterion(output,random_label_tensor)
        #stn_loss.backward()
        #total_loss = stn_loss + loss
        #total_loss = loss
        loss.backward()
        optimizer2.step()
        #acc = Accuracy(x.data,label.data)

        #stn_losses.update(stn_loss.item(), net_input.size(0))
        #losses.update(loss.item(), res_input.size(0))

        losses += loss.item()
        train_correct_sample += (predicted_label == label).cpu().sum().numpy() #acc
        #stn_losses += stn_loss.item()

    losses /= len(train_dataloader.dataset)
    train_correct_sample/= len(train_dataloader.dataset)
    ##total_losses = losses 
    #print(total_losses)

        #_, predicted1 = torch.max(x1.data,1)
        #stn_total += T1.size(0)
        #stn_acc1 = (x1 == T1).sum

        #_, predicted2 = torch.max(x2.data,1)
        #stn_acc2 = (x2 == T2).sum

        #_, predicted3 = torch.max(x3.data,1)
        #stn_acc3 = (x3 == T3).sum
 
        #stn_correct += (stn_acc1 + stn_acc2 + stn_acc3)/3
        #stn_acc = 100*stn_correct / stn_total

        #order_losses.update(loss.item(), res_input.size(0))


        #_, pred1 = torch.max(I1.data,1)
        #_, pred2 = torch.max(I2.data,1)
        #_, pred3 = torch.max(I3.data,1)
        #_, pred4 = torch.max(I4.data,1)
        #_, pred5 = torch.max(I5.data,1)
        #_, pred6 = torch.max(I6.data,1)
        #order_total += target.size(0)
        #order_acc1 = (I1 == target).sum
        #order_acc2 = (I2 == target).sum
        #order_acc3 = (I3 == target).sum
        #order_acc4 = (I4 == target).sum
        #order_acc5 = (I5 == target).sum
        #order_acc6 = (I6 == target).sum
        #order_correct += order_acc1*output1+order_acc2*output2+order_acc3*output3+order_acc4*output4+order_acc5*output5+order_acc6*output6
        #order_acc = 100*order_correct / order_total


   # return total_losses,stn_losses,losses,input_1,input_2,input_3,x1,x2,x3,random_x1,random_x2,random_x3,T1,T2,T3,I1,I2,I3,I4,I5,I6,targets,output,random_label_tensor
    return train_correct_sample,losses,T1,T2,T3,I1,I2,I3,I4,I5,I6,targets,prob,predicted_label,random_label_tensor
    #return losses.avg


#tensorboard --logdir=./logs
            #writer = SummaryWriter('./logs')
            #writer.add_scalar("train_loss",loss,epoch)        
            #writer.close()
            #if epoch %1 ==0:
             #   print('epoch',epoch,'Train_loss',loss.item())


def evaluate(net2,test_dataloader,batchsize,device):
#def evaluate(net,test_dataloader,device):

    with torch.no_grad():
        #使用评估模式
        #net1.eval()
        net2.eval()

        #criterion = nn.MSELoss().to(device)
        criterion_stn = nn.MSELoss().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        #eval_stn_losses = 0
        #eval_order_losses = 0
        train_correct_sample=0

        eval_stn_losses = 0
        eval_order_losses = 0


        for i,data in enumerate(test_dataloader):

            res_input,input1,input2,input3,T1,T2,T3,targets,label,random_label_tensor = data


            res_input,targets,label,random_label_tensor= res_input.to(device),targets.to(device),label.to(device,dtype=torch.long),random_label_tensor.to(device,dtype=torch.float)
        #print(label)
            input1,input2,input3,T1,T2,T3 = input1.to(device), input2.to(device), input3.to(device), T1.to(device), T2.to(device), T3.to(device)
            #res_input,input1,input2,input3,T1,T2,T3,targets,label = data  #dataloader's output concat(3img),img1,img2,img3,gt1,gt2,gt3,gt_order
            #res_input,targets,label= res_input.to(device),targets.to(device),label.to(device)
            #input1,input2,input3,T1,T2,T3 = input1.to(device), input2.to(device), input3.to(device), T1.to(device), T2.to(device), T3.to(device)


            #theta_1,theta_2,theta_3,input_1,input_2,input_3,x1,x2,x3,res_input = net1(net_input) #make concat(3img) into STN and return 3 theta ,3 input_image,3 affine image and concat
            output,prob = net2(res_input) # concat 3 affine image into order network


#####mask image of STN's output#####
            #mask_rice_tensor = (random_x1 >0).float()
            #mask_op1_tensor = (random_x2 >0).float()
            #mask_op2_tensor = (random_x3 >0).float()

#####get six image about composition#####
            #rice op1 op2
            #img1_hollow1 = torch.mul(random_x1,mask_op1_tensor) #第一层挖空
            #img1_bg1_hollow1 = random_x1-img1_hollow1 #第一层+挖空
            #img1_bg2 = img1_bg1_hollow1 + random_x2#第二层
            #img1_hollow2 = torch.mul(img1_bg2,mask_op2_tensor)#第二层挖空
            #img1_bg2_hollow2 = img1_bg2 - img1_hollow2#第二层+挖空
            #I1 = img1_bg2_hollow2 + random_x3
            #rice op2 op1
            #img2_hollow1 = torch.mul(random_x1,mask_op2_tensor) #第一层挖空
            #img2_bg1_hollow1 = random_x1-img2_hollow1 #第一层+挖空
            #img2_bg2 = img2_bg1_hollow1 + random_x3#第二层
            #img2_hollow2 = torch.mul(img2_bg2,mask_op1_tensor)#第二层挖空
            #img2_bg2_hollow2 = img2_bg2 - img2_hollow2#第二层+挖空
            #I2 = img2_bg2_hollow2 + random_x2

            #op1 rice op2
            #img3_hollow1 = torch.mul(random_x2,mask_rice_tensor) #第一层挖空
            #img3_bg1_hollow1 = random_x2-img3_hollow1 #第一层+挖空
            #img3_bg2 = img3_bg1_hollow1 + random_x1#第二层
            #img3_hollow2 = torch.mul(img3_bg2,mask_op2_tensor)#第二层挖空
            #img3_bg2_hollow2 = img3_bg2 - img3_hollow2#第二层+挖空
            #I3 = img3_bg2_hollow2 + random_x3

            #op1 op2 rice
            #img4_hollow1 = torch.mul(random_x2,mask_op2_tensor) #第一层挖空
            #img4_bg1_hollow1 = random_x2-img4_hollow1 #第一层+挖空
            #img4_bg2 = img4_bg1_hollow1 + random_x3#第二层
            #img4_hollow2 = torch.mul(img4_bg2,mask_rice_tensor)#第二层挖空
            #img4_bg2_hollow2 = img4_bg2 - img4_hollow2#第二层+挖空
            #I4 = img4_bg2_hollow2 + random_x1
            #op2 rice op1
            #img5_hollow1 = torch.mul(random_x3,mask_rice_tensor) #第一层挖空
            #img5_bg1_hollow1 = random_x3-img5_hollow1 #第一层+挖空
            #img5_bg2 = img5_bg1_hollow1 + random_x1#第二层
            #img5_hollow2 = torch.mul(img5_bg2,mask_op1_tensor)#第二层挖空
            #img5_bg2_hollow2 = img5_bg2 - img5_hollow2#第二层+挖空
            #I5 = img5_bg2_hollow2 + random_x2
            #op2 op1 rice
            #img6_hollow1 = torch.mul(random_x3,mask_op1_tensor) #第一层挖空
            #img6_bg1_hollow1 = random_x3-img6_hollow1 #第一层+挖空
            #img6_bg2 = img6_bg1_hollow1 + random_x2#第二层
            #img6_hollow2 = torch.mul(img6_bg2,mask_rice_tensor)#第二层挖空
            #img6_bg2_hollow2 = img6_bg2 - img6_hollow2#第二层+挖空
            #I6 = img6_bg2_hollow2 + random_x1


#####mask image of STN's output#####
            mask_rice_tensor = (T1 >0).float()
            mask_op1_tensor = (T2 >0).float()
            mask_op2_tensor = (T3 >0).float()

#####get six image about composition#####
            #rice op1 op2
            img1_hollow1 = torch.mul(T1,mask_op1_tensor) #第一层挖空
            img1_bg1_hollow1 = T1-img1_hollow1 #第一层+挖空
            img1_bg2 = img1_bg1_hollow1 + T2#第二层
            img1_hollow2 = torch.mul(img1_bg2,mask_op2_tensor)#第二层挖空
            img1_bg2_hollow2 = img1_bg2 - img1_hollow2#第二层+挖空
            I1 = img1_bg2_hollow2 + T3
            #rice op2 op1
            img2_hollow1 = torch.mul(T1,mask_op2_tensor) #第一层挖空
            img2_bg1_hollow1 = T1-img2_hollow1 #第一层+挖空
            img2_bg2 = img2_bg1_hollow1 + T3#第二层
            img2_hollow2 = torch.mul(img2_bg2,mask_op1_tensor)#第二层挖空
            img2_bg2_hollow2 = img2_bg2 - img2_hollow2#第二层+挖空
            I2 = img2_bg2_hollow2 + T2

            #op1 rice op2
            img3_hollow1 = torch.mul(T2,mask_rice_tensor) #第一层挖空
            img3_bg1_hollow1 = T2-img3_hollow1 #第一层+挖空
            img3_bg2 = img3_bg1_hollow1 + T1#第二层
            img3_hollow2 = torch.mul(img3_bg2,mask_op2_tensor)#第二层挖空
            img3_bg2_hollow2 = img3_bg2 - img3_hollow2#第二层+挖空
            I3 = img3_bg2_hollow2 + T3

            #op1 op2 rice
            img4_hollow1 = torch.mul(T2,mask_op2_tensor) #第一层挖空
            img4_bg1_hollow1 = T2-img4_hollow1 #第一层+挖空
            img4_bg2 = img4_bg1_hollow1 + T3#第二层
            img4_hollow2 = torch.mul(img4_bg2,mask_rice_tensor)#第二层挖空
            img4_bg2_hollow2 = img4_bg2 - img4_hollow2#第二层+挖空
            I4 = img4_bg2_hollow2 + T1
            #op2 rice op1
            img5_hollow1 = torch.mul(T3,mask_rice_tensor) #第一层挖空
            img5_bg1_hollow1 = T3-img5_hollow1 #第一层+挖空
            img5_bg2 = img5_bg1_hollow1 + T1#第二层
            img5_hollow2 = torch.mul(img5_bg2,mask_op1_tensor)#第二层挖空
            img5_bg2_hollow2 = img5_bg2 - img5_hollow2#第二层+挖空
            I5 = img5_bg2_hollow2 + T2
            #op2 op1 rice
            img6_hollow1 = torch.mul(T3,mask_op1_tensor) #第一层挖空
            img6_bg1_hollow1 = T3-img6_hollow1 #第一层+挖空
            img6_bg2 = img6_bg1_hollow1 + T2#第二层
            img6_hollow2 = torch.mul(img6_bg2,mask_rice_tensor)#第二层挖空
            img6_bg2_hollow2 = img6_bg2 - img6_hollow2#第二层+挖空
            I6 = img6_bg2_hollow2 + T1

#####STN loss#####
            #stn_loss1 = criterion_stn(x1,T_stn1_tensor)
            #stn_loss2 = criterion_stn(x2,T_stn2_tensor)
            #stn_loss3 = criterion_stn(x3,T_stn3_tensor)
            #stn_loss = stn_loss1 + stn_loss2 + stn_loss3

        #train_acc.update(acc.item(),inputs.size(0))

#####get P1~p6 from order network##### 
            #index1=torch.LongTensor([[0]]).to(device)
            #output1=torch.gather(output,1,index1)
            #index2=torch.LongTensor([[1]]).to(device)
            #output2=torch.gather(output,1,index2)
            #index3=torch.LongTensor([[2]]).to(device)
            #output3=torch.gather(output,1,index3)
            #index4=torch.LongTensor([[3]]).to(device)
            #output4=torch.gather(output,1,index4)
            #index5=torch.LongTensor([[4]]).to(device)
            #output5=torch.gather(output,1,index5)
            #index6=torch.LongTensor([[5]]).to(device)
            #output6=torch.gather(output,1,index6)

######order loss#####
            #loss1 = criterion(I1, targets)
            #loss2 = criterion(I2, targets)
            #loss3 = criterion(I3, targets)
            #loss4 = criterion(I4, targets)
            #loss5 = criterion(I5, targets)
            #loss6 = criterion(I6, targets)
            #loss = output1*loss1+output2*loss2+output3*loss3+output4*loss4+output5*loss5+output6*loss6

            ##loss = criterion(total_image, targets)
            #eval_stn_losses.update(stn_loss.item(), net_input.size(0))
            #eval_order_losses.update(loss.item(), res_input.size(0))

            loss = criterion(prob,random_label_tensor)
            _, predicted_label = torch.max(output, 1)
            #loss = criterion(output,random_label_tensor)
            #eval_stn_losses += stn_loss.item()
            eval_order_losses +=loss.item()
            train_correct_sample += (predicted_label == label).cpu().sum().numpy()
        #eval_stn_losses /= len(test_dataloader.dataset)
        eval_order_losses /= len(test_dataloader.dataset)
        train_correct_sample /= len(test_dataloader.dataset)
        #total_loss = eval_stn_losses + eval_order_losses
        ###total_loss = eval_order_losses
            #eval_stn_losses.update(stn_loss.item(), net_input.size(0))
            #eval_order_losses.update(loss.item(), res_input.size(0))
            #eval_stn_losses += stn_loss.item()
            #eval_order_losses += loss.item()

            #_, predicted1 = torch.max(x1.data,1)
            #eval_stn_total += T1.size(0)
            #eval_stn_acc1 = (x1 == T1).sum

            #_, predicted2 = torch.max(x2.data,1)
            #eval_stn_acc2 = (x2 == T2).sum

            #_, predicted3 = torch.max(x3.data,1)
            #eval_stn_acc3 = (x3 == T3).sum
 
            #eval_stn_correct += (aeval_stn_cc1 + eval_stn_acc2 + eval_stn_acc3)/3
            #eval_stn_acc = 100*stn_correct / stn_total



    
            #eval_loss += loss.item()
            #pred_label = pred.max(1,keepdim=True)[1]
            #eval_acc += pred_label.eq(label.view_as(pred_label)
            #).sum().item()


       # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #      .format(eval_loss, eval_acc, len(test_dataloader.dataset),
         #             100. * eval_acc / len(test_dataloader.dataset)))
           # if batch_idx % batchsize == 0:
            #    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
             #         .format(eval_loss, eval_acc, len(test_dataloader.dataset),
                  #            100. * eval_acc / len(test_dataloader.dataset)))
            #losses.update(eval_loss, net_input.size(0))
    return train_correct_sample,eval_order_losses,T1,T2,T3,I1,I2,I3,I4,I5,I6,targets,prob,predicted_label,random_label_tensor

if __name__ == "__main__":
    main()
