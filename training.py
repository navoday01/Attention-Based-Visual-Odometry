import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import MyDataset, vec_to_pose, pose_to_vec, find_abs
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from models import AttNet
import torch.nn as nn
import time
import sys

device = 'cuda'


if(len(sys.argv)==1):
  print("Path of the dataset is not given")
  exit()
elif(len(sys.argv)==2):
  print("Scene was not selected, heads scene is selected by default")  

data_dir = sys.argv[1]

if(len(sys.argv)==2):
  scene = 'heads'
else:
  scene =  sys.argv[2] 


num_workers = 1
writer = SummaryWriter()



model = AttNet().to(device)
# model.load_state_dict(torch.load("Model"))

rgb_transform = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[130, 115, 110],
    #   std=[60, 60, 60])
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

d_transform = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.022],
      std=[0.03])
  ])
# aa = transforms.ToPILImage()


optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

mse = nn.MSELoss()
mae = nn.L1Loss()
hloss = nn.HuberLoss()
E = 5000 # epochs
prev_loss = 1e10
# prev_loss = 17


hop = 25
train_dataset = MyDataset(data_dir,scene,hop,device,train = True,test = False,
                  rgb_transform=rgb_transform,d_transform=d_transform)
val_dataset = MyDataset(data_dir,scene,hop,device,train = False,test = False,
                  rgb_transform=rgb_transform,d_transform=d_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                  num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                  num_workers=num_workers)


############# Training #######################

def train(t,model,optimizer,train_loader):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    l = []
    l1 = []
    l2 = []
    for idx, (rgb_image,d_image,pose) in enumerate(train_loader):
        # print(rgb_image.shape)
        # print(d_image.shape)
        # print(pose.shape)
        # if(rgb_image.shape[0]<26):
        #   sys.stdout.write('\r'+str(idx+1)+"/"+str(len(train_loader))+" completed, Training Loss: "+str(format(loss.item(),".2f")))
        #   continue
        rgb_image = rgb_image[0].to(device)
        d_image = d_image[0].to(device)
        d_image = torch.unsqueeze(d_image,dim = 1)
        # print(d_image.shape)
        # print(rgb_image.shape)
        T_i_in_o = pose[0,-2].to(device)
        T_f_in_o = pose[0,-1].to(device)
        pose = pose[0,-1,:3].to(device)
        # print(pose.shape)
        # print(T_i_in_o.shape, T_f_in_o.shape)
        pose = torch.linalg.inv(T_i_in_o)@T_f_in_o
        pose = pose[:3]

        pose = pose.float()
        rgb_image = rgb_image.float()
        d_image = d_image.float()
        
        pose_vec_pred = model(rgb_image,d_image)
        pose_vec = pose_to_vec(pose)
        pose_pred = vec_to_pose(pose_vec_pred)

        loss1 = find_abs(pose_vec_pred[:3],pose_vec[:3])
        loss2 = find_abs(pose_vec_pred[3:],pose_vec[3:])
        loss = loss1 + 100*loss2

        sys.stdout.write('\r'+str(idx+1)+"/"+str(len(train_loader))+" completed, Training Loss: "+str(format(loss.item(),".2f"))+" L1: "+str(format(loss1.item(),".2f"))+" L2: "+str(format(loss2.item(),".2f")))
        l.append(loss.item())
        l1.append(loss1.item())
        l2.append(loss2.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(l)/len(l), sum(l1)/len(l1), sum(l2)/len(l2)


def val(t,model,val_loader):
    model.eval()
    model.zero_grad()
    l = []
    l1 = []
    l2 = []
    for idx, (rgb_image,d_image,pose) in enumerate(val_loader):
        # if(rgb_image.shape[0]<26):
        #   sys.stdout.write('\r'+str(idx+1)+"/"+str(len(val_loader))+" completed, Validation Loss: "+str(format(loss.item(),".2f")))
        #   continue
        rgb_image = rgb_image[0].to(device)
        d_image = d_image[0].to(device)
        d_image = torch.unsqueeze(d_image,dim = 1)
        # print(d_image.shape)
        # print(rgb_image.shape)
        T_i_in_o = pose[0,-2].to(device)
        T_f_in_o = pose[0,-1].to(device)
        pose = pose[0,-1,:3].to(device)
        # print(pose.shape)
        # print(T_i_in_o.shape, T_f_in_o.shape)
        pose = torch.linalg.inv(T_i_in_o)@T_f_in_o
        pose = pose[:3]
        
        pose_vec_pred = model(rgb_image,d_image)
        pose_vec = pose_to_vec(pose)
        pose_pred = vec_to_pose(pose_vec_pred)

        loss1 = find_abs(pose_vec_pred[:3],pose_vec[:3])
        loss2 = find_abs(pose_vec_pred[3:],pose_vec[3:])
        loss = loss1 + 100*loss2

        sys.stdout.write('\r'+str(idx+1)+"/"+str(len(val_loader))+" completed, Validation Loss: "+str(format(loss.item(),".2f"))+" L1: "+str(format(loss1.item(),".2f"))+" L2: "+str(format(loss2.item(),".2f")))
        l.append(loss.item())
        l1.append(loss1.item())
        l2.append(loss2.item())
    return sum(l)/len(l), sum(l1)/len(l1), sum(l2)/len(l2)


##############################################

################ Saving Models ###############

def save_models(tl,pr = False):
    global prev_loss
    if (tl<prev_loss):
        if(pr == True):
            print('Validation Loss decreased, saving models')
        torch.save(model.state_dict(),'Model21temp')
        prev_loss = tl


##############################################


############## Learning the functions ########

for t in range(E):
    print('Epoch:',t)
    t1 = time.time()
    tl, tl1, tl2 = train(t,model,optimizer,train_loader)
    print(' ')
    val_tl, val_tl1, val_tl2 = val(t,model,val_loader)
    writer.add_scalar("Training Loss",tl,t)
    writer.add_scalar("Validation Loss",val_tl,t)
    writer.add_scalar("Training Position Loss",tl1,t)
    writer.add_scalar("Training Rotation Loss",tl2,t)
    writer.add_scalar("Validation Position Loss",val_tl1,t)
    writer.add_scalar("Validation Rotation Loss",val_tl2,t)
    t2 = time.time()
    print(' ')
    print('Avg. Training Loss =', format(tl,".2f"),'Position Training Loss =', format(tl1,".2f"),'Rotation Training Loss =', format(tl2,".2f"),'Avg. Validation Loss =', format(val_tl,".2f"),'Position Validation Loss =', format(val_tl1,".2f"), 'Rotation Validation Loss =', format(val_tl2,".2f"),'Time = %.2f' %(t2-t1),'s')
    save_models(val_tl,True)
writer.flush()

