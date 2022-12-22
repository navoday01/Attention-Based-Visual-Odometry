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
import pandas as pd

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
pred_traj_x = []
pred_traj_y = []
pred_traj_z = []
target_traj_x = []
target_traj_y = []
target_traj_z = []

model = AttNet().to(device)
model.load_state_dict(torch.load("Model2"))

rgb_transform = transforms.Compose([
   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]),
      transforms.Resize((480,640))
  ])

d_transform = transforms.Compose([
 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.022],
      std=[0.03])
  ,transforms.Resize((480,640))])


mse = nn.MSELoss()
mae = nn.L1Loss()
hloss = nn.HuberLoss()

hop = 1
test_dataset = MyDataset(data_dir,scene,hop,device,train = False,test = True,
                  rgb_transform=rgb_transform,d_transform=d_transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                  num_workers=num_workers)



############# Testing #######################
def test(model,test_loader):
    global pred_traj_x
    global pred_traj_y
    global pred_traj_z
    global target_traj_x
    global target_traj_y
    global target_traj_z
    model.eval()
    model.zero_grad()
    l = []
    lt = []
    la = []
    for idx, (rgb_image,d_image,pose) in enumerate(test_loader):
      
        rgb_image = rgb_image[0].to(device)
        d_image = d_image[0].to(device)
        d_image = torch.unsqueeze(d_image,dim = 1)
      
        T_i_in_o = pose[0,-2].to(device)
        T_f_in_o = pose[0,-1].to(device)
        pose = pose[0,-1,:3].to(device)
      
        pose = torch.linalg.inv(T_i_in_o)@T_f_in_o
        pose = pose[:3]
        
        print(rgb_image.isnan().any())
        print(d_image.isnan().any())

        pose_vec_pred = model(rgb_image,d_image)       
        pose_vec = pose_to_vec(pose)

        pose_pred = vec_to_pose(pose_vec_pred)
        T_f_in_i_pred = torch.vstack([pose_pred,torch.Tensor([[0,0,0,1]]).to(device)])
        T_f_in_o_pred = T_i_in_o@T_f_in_i_pred
            
        predicted_pose = pose_to_vec(T_f_in_o_pred).cpu().detach().numpy()
       
        ground_truth_pose = pose_to_vec(T_f_in_o).cpu().detach().numpy()
        pred_traj_x.append(predicted_pose[0])
        pred_traj_y.append(predicted_pose[1])
        pred_traj_z.append(predicted_pose[2])
        target_traj_x.append(ground_truth_pose[0])
        target_traj_y.append(ground_truth_pose[1])
        target_traj_z.append(ground_truth_pose[2])

        pose_vec_pred = pose_vec_pred.to(torch.float16)
        

       
        loss_t = find_abs(pose_vec_pred[:3],pose_vec[:3])
        loss_a = find_abs(pose_vec_pred[3:],pose_vec[3:])

       
        loss = mse(pose_pred,pose)

        sys.stdout.write('\r'+str(idx+1)+"/"+str(len(test_loader))+" completed, Test Loss: "+str(format(loss.item(),".2f"))+", Translation Loss: "+str(format(loss_t.item(),".2f"))+", Angle Loss: "+str(format(loss_a.item(),".2f")))

        if(pose_vec_pred.isnan().any()==0):
          lt.append(loss_t.item())
          la.append(loss_a.item())
          l.append(loss.item())

    
    df = pd.DataFrame({'pred_x':pred_traj_x, 'pred_y':pred_traj_y, 'pred_z':pred_traj_z, 'true_x':target_traj_x, 'true_y':target_traj_y, 'true_z':target_traj_z})
    df.to_csv(str(scene)+'.csv', index = False)
    return sum(l)/len(l), sum(lt)/len(lt), sum(la)/len(la)


##############################################

############## Calling test ########

tl, tlt, tla = test(model,test_loader)
print(' ')
print('Avg. testing Loss =', format(tl,".2f"),'Avg. Translation Loss =', format(tlt,".2f"),'Avg. Angle Loss =', format(tla,".2f"))
