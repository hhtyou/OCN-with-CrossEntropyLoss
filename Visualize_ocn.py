import os
import torch,torchvision
import matplotlib.pyplot as plt
from TestLoader import get_testloader,tensor_to_array
from torchvision.transforms import ToTensor,ToPILImage
from Net import STN_Net,Resnet
from config import parse_args
#from Test import get_testloader
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage

args = parse_args()
if args.use_gpu and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


net2 = Resnet().to(device)
#net = STN_Net(args.use_stn).to(device)

path_checkpoint2 = "finalcorss6result/checkpoint/net2_epoch_258.pth"

checkpoint2 = torch.load(path_checkpoint2)


net2.load_state_dict(checkpoint2['net2'])
##net2.load_state_dict(checkpoint2)

#train_dataloader,test_loader = get_dataloader(args.test_batch_size)
test_loader = get_testloader(args.test_batch_size)
#test_loader = get_dataloader(args.test_batch_size)
tensor_to_image = ToPILImage()
with torch.no_grad():
    net2.eval()
    for i,data in enumerate(test_loader):
        res_input,input1,input2,input3,T1,T2,T3,targets = data  #dataloader's output concat(3img),img1,img2,img3,gt1,gt2,gt3,gt_order
        res_input,targets= res_input.to(device),targets.to(device),

        input1,input2,input3,T1,T2,T3 = input1.to(device), input2.to(device), input3.to(device), T1.to(device), T2.to(device), T3.to(device)
        output,prob = net2(res_input)
        #net_input,input1,input2,input3,T1,T2,T3,mask_input1 = data
        #print(inputs.shape)
        #net_input = net_input.to(device)
        #input1,input2,input3,T1,T2,T3 = input1.to(device), input2.to(device), input3.to(device), T1.to(device), T2.to(device), T3.to(device)
        #mask_input1=mask_input1.to(device)
        #theta_1,theta_2,theta_3,input_1,input_2,input_3 = net(net_input)  #输出是theta
        #print(input_1.shape)

        #grid_1 = F.affine_grid(theta_1, input_1.size(),align_corners=True)
        #grid_2 = F.affine_grid(theta_2, input_2.size(),align_corners=True)
        #grid_3 = F.affine_grid(theta_3, input_2.size(),align_corners=True)
        #grid_4 = F.affine_grid(theta_1, mask_input1.size(),align_corners=True)
        #x1 = F.grid_sample(input_1, grid_1, padding_mode="border",align_corners=True)
        #x2 = F.grid_sample(input_2, grid_2, padding_mode="border",align_corners=True)
        #x3 = F.grid_sample(input_3, grid_3, padding_mode="border",align_corners=True)
        #x4 = F.grid_sample(mask_input1, grid_4, padding_mode="border",align_corners=True)
        #print(x4.shape)
        #x1_mask = (x1 >0).float()


        #根据输入图片计算变换后图片位置填充的像素值
    #output_tensor = F.grid_sample(data,grid,align_corners=True)
    if not os.path.isdir('./TEST_img'):
        os.makedirs('./TEST_img')

    #print(theta_2)
    #print(input1.shape)
    tensor_to_image = ToPILImage()

    T1=T1[0,...]
    #T1 = torch.squeeze(T1,dim=0)
    T_data1 =tensor_to_image(T1)
    T_data1.save('TEST_img/T1.png')

    T2=T2[0,...]
    #T2 = torch.squeeze(T2,dim=0)
    T_data2 =tensor_to_image(T2)
    T_data2.save('TEST_img/T2.png')

    T3=T3[0,...]
    #T3 = torch.squeeze(T3,dim=0)
    T_data3 =tensor_to_image(T3)
    T_data3.save('TEST_img/T3.png')

    targets=targets[0,...]
    #target = torch.squeeze(targets,dim=0)
    target =tensor_to_image(targets)
    target.save('TEST_img/target.png')
    
    with open(os.path.join('TEST_img','prob.txt'),"w") as f:
        f.write(str(prob))

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

    I1 =tensor_to_image(I1)
    I1.save('TEST_img/I1.png')

    I2 =tensor_to_image(I2)
    I2.save('TEST_img/I2.png')

    I3 =tensor_to_image(I3)
    I3.save('TEST_img/I3.png')

    I4 =tensor_to_image(I4)
    I4.save('TEST_img/I4.png')

    I5 =tensor_to_image(I5)
    I5.save('TEST_img/I5.png')

    I6 =tensor_to_image(I6)
    I6.save('TEST_img/I6.png')
