import torch
import torch.nn as nn
from pytorch_revgrad import RevGrad
import random
import numpy as np
from torch.serialization import load
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


class Model(nn.Module):
    def __init__(self, grl=False):
        super().__init__()
        self.grl = grl
        self.layer1 = nn.Linear(2,1)
        self.layer2 = nn.Linear(1,1)
    
    def forward(self, x):
        x1 = self.layer1(x)
        if self.grl:
            RevGrad()
        x2 = self.layer2(x)
        return x2

model = Model()
model.load_state_dict(load("model.pt"))
for param in model.parameters():
    print(param.grad)
x = torch.Tensor([1,2])
loss = model(x)
loss.backward()
for param in model.parameters():
    print(param.grad)

model = Model(grl=True)
model.load_state_dict(load("model.pt"))
for param in model.parameters():
    print(param.grad)
x = torch.Tensor([1,2])
loss = model(x)
loss.backward()
for param in model.parameters():
    print(param.grad)