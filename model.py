''' 
This file is mostly a copy of https://github.com/aerorobotics/neural-fly/blob/main/mlmodel.py
There are some simplifications
'''

import collections
import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.DoubleTensor')

Model = collections.namedtuple('Model', 'phi h options')

class Phi_Net(nn.Module):
    def __init__(self, dim_x, dim_a):
        super(Phi_Net, self).__init__()

        self.dim_x = dim_x
        self.dim_a = dim_a
        self.fc1 = nn.Linear(dim_x, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(50, dim_a-1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for training
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)

# Cross-entropy loss
class H_Net_CrossEntropy(nn.Module):
    def __init__(self, dim_a, num_c):
        super(H_Net_CrossEntropy, self).__init__()
        self.dim_a = dim_a
        self.num_c = num_c

        self.fc1 = nn.Linear(dim_a, 20)
        self.fc2 = nn.Linear(20, num_c)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(*, phi_net, h_net, modelname):
    if not os.path.isdir('./models/'):
        os.makedirs('./models/')
    if h_net is not None:
        torch.save({
            'phi_net_state_dict': phi_net.state_dict(),
            'h_net_state_dict': h_net.state_dict(),
            'dim_x': phi_net.dim_x,
            'dim_a': phi_net.dim_a,
            'num_c': h_net.num_c,
        }, './models/' + modelname + '.pth')
    else:
        torch.save({
            'phi_net_state_dict': phi_net.state_dict(),
            'h_net_state_dict': None,
            'dim_x': phi_net.dim_x,
            'dim_a': phi_net.dim_a,
            'num_c': None,
        }, './models/' + modelname + '.pth')

def load_model(model_path):
    model = torch.load(model_path)

    phi_net = Phi_Net(model.get('dim_x'), model.get('dim_a'))
    # h_net = H_Net_CrossEntropy(options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])
    # h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    # h_net.eval()

    return phi_net