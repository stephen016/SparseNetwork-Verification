# A simple 3 layer MLP model only contains ReLU activation

import torch
import torch.nn as nn
import numpy as np

from models.Pruneable import Pruneable

class MLP3(Pruneable):
    def __init__(self,device="cuda",hidden_dim=64,output_dim=10,input_dim=(1,),**kwargs):
        super(MLP3,self).__init__(device=device,output_dim=output_dim,input_dim=input_dim,**kwargs)
        self.hidden_dim = hidden_dim[0]
        self.input_dim = int(np.prod(input_dim))
        self.output_dim = output_dim
        gain=1
        self.layers = nn.Sequential(
            self.Linear(input_dim=input_dim,output_dim=hidden_dim,bias=True,gain=gain),
            nn.ReLU(),
            self.Linear(input_dim=hidden_dim,output_dim=hidden_dim,bias=True,gain=gain),
            nn.ReLU(),
            self.Linear(input_dim=hidden_dim,output_dim=output_dim,bias=True,gain=gain)
        ).to(device)
        
    
    def forward(self,x:torch.Tensor):
        x = x.view(x.shape[0],-1)
        return self.layers.forward(x)
        