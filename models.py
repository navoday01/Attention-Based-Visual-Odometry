import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_printoptions(precision = 8)


class NetGated(nn.Module):
  def __init__(self):
    super().__init__()
    # for rgb images
    self.cc11 = nn.Conv2d(3,16,kernel_size=3, padding='same')
    self.cc12 = nn.Conv2d(16,16,kernel_size=3, padding='same')
    self.cm1 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.cc21 = nn.Conv2d(16,32,kernel_size=3, padding='same')
    self.cc22 = nn.Conv2d(32,32,kernel_size=3, padding='same')
    self.cm2 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.cc31 = nn.Conv2d(32,48,kernel_size=3, padding='same')
    self.cc32 = nn.Conv2d(48,48,kernel_size=3, padding='same')
    self.cm3 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.cc41 = nn.Conv2d(48,64,kernel_size=3, padding='same')
    self.cc42 = nn.Conv2d(64,64,kernel_size=3, padding='same')
    self.cm4 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.cfc = nn.Linear(76800,512)

    # for depth images
    self.dc11 = nn.Conv2d(1,16,kernel_size=3, padding='same')
    self.dc12 = nn.Conv2d(16,16,kernel_size=3, padding='same')
    self.dm1 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.dc21 = nn.Conv2d(16,32,kernel_size=3, padding='same')
    self.dc22 = nn.Conv2d(32,32,kernel_size=3, padding='same')
    self.dm2 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.dc31 = nn.Conv2d(32,48,kernel_size=3, padding='same')
    self.dc32 = nn.Conv2d(48,48,kernel_size=3, padding='same')
    self.dm3 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.dc41 = nn.Conv2d(48,64,kernel_size=3, padding='same')
    self.dc42 = nn.Conv2d(64,64,kernel_size=3, padding='same')
    self.dm4 = nn.MaxPool2d(kernel_size = 2,stride = 2)

    self.dfc = nn.Linear(76800,512)

    self.fc1 = nn.Linear(1024,64)
    self.fc2 = nn.Linear(64,2)

  def rgb_conv(self,c):
    c = F.relu(self.cc11(c))
    c = F.relu(self.cc12(c))
    c = self.cm1(c)

    c = F.relu(self.cc21(c))
    c = F.relu(self.cc22(c))
    c = self.cm2(c)

    c = F.relu(self.cc31(c))
    c = F.relu(self.cc32(c))
    c = self.cm3(c)

    c = F.relu(self.cc41(c))
    c = F.relu(self.cc42(c))
    c = self.cm4(c)

    c = torch.flatten(c, start_dim = 1)
    c = F.relu(self.cfc(c))
    return c

  def depth_conv(self,d):
    d = F.relu(self.dc11(d))
    d = F.relu(self.dc12(d))
    d = self.dm1(d)

    d = F.relu(self.dc21(d))
    d = F.relu(self.dc22(d))
    d = self.dm2(d)

    d = F.relu(self.dc31(d))
    d = F.relu(self.dc32(d))
    d = self.dm3(d)

    d = F.relu(self.dc41(d))
    d = F.relu(self.dc42(d))
    d = self.dm4(d)

    d = torch.flatten(d, start_dim = 1)
    d = F.relu(self.dfc(d))
    return d

  def forward(self,c,d):
    c = self.rgb_conv(c)
    d = self.depth_conv(d)
    
    x = torch.cat([c,d],dim = -1)
    x = self.fc1(x)
    x = self.fc2(x)

    x = x[:,:1]*c + x[:,1:]*d
    
    return x


class TempAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(12800,512)
    self.fc2 = nn.Linear(1024,25)
    self.fc3 = nn.Linear(1024,100)
    self.fc4 = nn.Linear(100,6)

  def forward(self,x):
    emb_hist = x[:25]
    emb = x[25]

    x1 = torch.flatten(emb_hist)
    x1 = self.fc1(x1)
    x1 = torch.cat([x1,emb])
    x1 = F.softmax(self.fc2(x1),dim = -1)
    # x1 = self.fc2(x1)
    # print(torch.argmax(x1))
    # print(x1)
    x1 = x1@emb_hist
    x1 = torch.cat([x1,emb])
    x1 = F.relu(self.fc3(x1))
    x1 = self.fc4(x1)
    return x1

class AttNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = NetGated()
    self.decoder = TempAttention()

  def forward(self,c,d):
    res = self.encoder(c,d)
    res = self.decoder(res)
    res[3:] = torch.pi*torch.tanh(res[3:])
    return res