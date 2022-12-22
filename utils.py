import os
import os.path as osp
import numpy as np
from torch.utils import data
# from torchvision.datasets.folder import default_loader
# from PIL import Image
import sys
import pickle
# import transforms3d.quaternions as txq
# import transforms3d.euler as txe
from torch import sin, cos
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '../')

# class MyDataset(data.Dataset):
#   def __init__(self,data_dir,scene,rgb_transform=None,d_transform=None, train = True):
#     self.data_dir = data_dir
#     self.rgb_transform = rgb_transform
#     self.d_transform = d_transform
#     self.rgb_filenames = []
#     self.d_filenames = []
#     self.p_filenames = []
#     base_dir = os.path.join(self.data_dir,scene)
    
#     if train:
#       split_file = osp.join(base_dir, 'TrainSplit.txt')
#     else:
#       split_file = osp.join(base_dir, 'TestSplit.txt')
#     with open(split_file, 'r') as f:
#       seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

#     for seq in seqs:
#       seq_dir = os.path.join(base_dir,'seq-{:02d}'.format(seq))
#       p_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
#                        n.find('pose') >= 0]
#       rgb_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
#                        n.find('color') >= 0]
#       d_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
#                        n.find('depth') >= 0]
      
#       self.p_filenames.extend(p_filenames)
#       self.rgb_filenames.extend(rgb_filenames)
#       self.d_filenames.extend(d_filenames)
#     self.p_filenames = sorted(self.p_filenames)
#     self.rgb_filenames = sorted(self.rgb_filenames)
#     self.d_filenames = sorted(self.d_filenames)

#   def __getitem__(self,index):
#     rgb_img = load_image(self.rgb_filenames[index])
#     d_img = load_image(self.d_filenames[index])
#     pose = load_pose(self.p_filenames[index])
    
#     if self.rgb_transform is not None:
#       rgb_img = self.rgb_transform(rgb_img)
    
#     if self.d_transform is not None:
#       d_img = self.d_transform(d_img)
    
#     return rgb_img,d_img,pose

#   def __len__(self):
#     return len(self.p_filenames)

class MyDataset(data.Dataset):
  def __init__(self,data_dir,scene,hop, device,rgb_transform=None,d_transform=None, train = True, test = False):
    self.data_dir = data_dir
    self.rgb_transform = rgb_transform
    self.d_transform = d_transform
    self.hop = hop
    self.rgb_filenames = []
    self.d_filenames = []
    self.p_filenames = []
    base_dir = os.path.join(self.data_dir,scene)
    
    if train:
      split_file = osp.join(base_dir, 'TrainSplit.txt')
    else:
      if test:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      else:
        split_file = osp.join(base_dir, 'ValSplit.txt')
    with open(split_file, 'r') as f:
      seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

    for seq in seqs:
      seq_dir = os.path.join(base_dir,'seq-{:02d}'.format(seq))
      p_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
      rgb_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('color') >= 0]
      d_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('depth') >= 0]
      
      self.p_filenames.extend(p_filenames)
      self.rgb_filenames.extend(rgb_filenames)
      self.d_filenames.extend(d_filenames)
    self.p_filenames = sorted(self.p_filenames)
    self.rgb_filenames = sorted(self.rgb_filenames)
    self.d_filenames = sorted(self.d_filenames)
    self.dummy_rgb_img = torch.zeros([3,480,640])
    self.dummy_d_img = torch.zeros([480,640])
    self.dummy_pose = torch.zeros([4,4])
    self.device = device

  def __getitem__(self,index):
    index = index*self.hop + np.random.randint(self.hop)
    if index==0:
      index += 1
    rgb_img = torch.empty([26,3,480,640])
    d_img = torch.empty([26,480,640])
    pose = torch.empty([26,4,4])
    k = 0
    for i in range(index-25,index+1):
      if(i<0):
        rgb_img[k] = self.dummy_rgb_img
        d_img[k] = self.dummy_d_img
        pose[k] = self.dummy_pose
      else:
        tr = load_image(self.rgb_filenames[i])
        if self.rgb_transform is not None:
          tr = self.rgb_transform(tr)
        rgb_img[k] = tr

        td = load_image(self.d_filenames[i])
        if self.d_transform is not None:
          td = self.d_transform(td)
        d_img[k] = td

        pose[k] = load_pose(self.p_filenames[i])
      k += 1
    
    # if self.rgb_transform is not None:
    #   rgb_img = self.rgb_transform(rgb_img)
    
    # if self.d_transform is not None:
    #   d_img = self.d_transform(d_img)
    # print(pose)
    return rgb_img,d_img,pose

  def __len__(self):
    return int(len(self.p_filenames)/self.hop)
    # return len(self.p_filenames)


class MyDataset2(data.Dataset):
  def __init__(self,data_dir,scene,hop, device,rgb_transform=None,d_transform=None, train = True, test = False):
    self.data_dir = data_dir
    self.rgb_transform = rgb_transform
    self.d_transform = d_transform
    self.hop = hop
    self.rgb_filenames = []
    self.d_filenames = []
    self.p_filenames = []
    base_dir = os.path.join(self.data_dir,scene)
    
    if train:
      split_file = osp.join(base_dir, 'TrainSplit.txt')
    else:
      if test:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      else:
        split_file = osp.join(base_dir, 'ValSplit.txt')
    with open(split_file, 'r') as f:
      seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

    for seq in seqs:
      seq_dir = os.path.join(base_dir,'seq-{:02d}'.format(seq))
      p_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
      rgb_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('color') >= 0]
      d_filenames = [os.path.join(seq_dir,n) for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('depth') >= 0]
      
      self.p_filenames.extend(p_filenames)
      self.rgb_filenames.extend(rgb_filenames)
      self.d_filenames.extend(d_filenames)
    self.p_filenames = sorted(self.p_filenames)
    self.rgb_filenames = sorted(self.rgb_filenames)
    self.d_filenames = sorted(self.d_filenames)
    self.dummy_rgb_img = torch.zeros([3,480,640])
    self.dummy_d_img = torch.zeros([480,640])
    self.dummy_pose = torch.zeros([4,4])
    self.device = device

  def __getitem__(self,index):
    index = np.max([index*self.hop + np.random.randint(self.hop),25])
    rgb_img = torch.empty([26,3,480,640])
    d_img = torch.empty([26,480,640])
    pose = torch.empty([26,4,4])
    k = 0
    for i in range(index-25,index+1):
      if(i<0):
        rgb_img[k] = self.dummy_rgb_img
        d_img[k] = self.dummy_d_img
        pose[k] = self.dummy_pose
      else:
        tr = load_image(self.rgb_filenames[i])
        if self.rgb_transform is not None:
          tr = self.rgb_transform(tr)
        rgb_img[k] = tr

        td = load_image(self.d_filenames[i])
        if self.d_transform is not None:
          td = self.d_transform(td)
        d_img[k] = td

        pose[k] = load_pose(self.p_filenames[i])
      k += 1
    
    # if self.rgb_transform is not None:
    #   rgb_img = self.rgb_transform(rgb_img)
    
    # if self.d_transform is not None:
    #   d_img = self.d_transform(d_img)
    # print(pose)
    return rgb_img,d_img,pose

  def __len__(self):
    return int(len(self.p_filenames)/self.hop)
    # return len(self.p_filenames)


def load_pose(filename):
  res = torch.Tensor(np.loadtxt(filename))
  return res


def load_image(filename):
  try:
    # img = loader(filename)
    # img = Image.open(filename)
    img = plt.imread(filename)
    # print(np.array(img).shape)
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  except:
    print('Could not load image {:s}, unexpected error'.format(filename))
    return None

  return img


def vec_to_pose(x):
  t = x[:3][...,None]
  roll = x[3]
  pitch = x[4]
  yaw = x[5]
  device = x.device

  Rz = torch.Tensor([[cos(yaw), -sin(yaw), 0],
                     [sin(yaw), cos(yaw), 0],
                     [0 , 0 , 1]]).to(device)

  Ry = torch.Tensor([[cos(pitch), 0, sin(pitch)],
                     [0, 1, 0],
                     [-sin(pitch) , 0 , cos(pitch)]]).to(device)

  Rx = torch.Tensor([[1, 0, 0],
                     [0, cos(roll), -sin(roll)],
                     [0 , sin(roll) , cos(roll)]]).to(device)

  R = Rz@Ry@Rx
  T = torch.cat([R,t],dim = 1)
  return T

def pose_to_vec(x):
  t = x[:3,3]
  R = x[:3,:3]
  pitch = -torch.arcsin(R[2,0])
  roll = torch.arctan2(R[2,1],R[2,2])
  yaw = torch.arctan2(R[1,0],R[0,0])
  res = torch.hstack([t,roll,pitch,yaw])
  return res


def find_abs(x,y):
  return torch.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)