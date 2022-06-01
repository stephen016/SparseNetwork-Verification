
from models.Pruneable import Pruneable

import numpy as np
import torch
import torch.nn as nn
class MLP5(Pruneable):
    def __init__(self,operation,resources,device="cuda",**kwargs):
        super(MLP5,self).__init__(device=device,operation=operation,resources=resources,**kwargs)
        gain=1
        modules=[]
        self.weight_list=[]
        self.bias_list=[]
        for ops,res in zip(operation,resources):
            if "Gemm" in ops:
                _out,_in = res['deeppoly'][0].shape
                modules.append(self.Linear(input_dim=_in,output_dim=_out,bias=True,gain=gain))
                modules.append(nn.ReLU())
                self.weight_list.append(torch.from_numpy(res['deeppoly'][0]))
                self.bias_list.append(torch.from_numpy(res['deeppoly'][1]))
        # set output_dim
        self.output_dim = _out
        # remove last relu
        modules = modules[0:-1]
        self.layers = nn.Sequential(*modules).to(device)
        
        self.init_weight(resources)
    
    def forward(self,x:torch.Tensor):
        x = x.view(x.shape[0],-1)
        return self.layers.forward(x)

    def init_weight(self,resources):
        states = self.state_dict()
        for i,state in enumerate(states):
            if "weight" in state:
                states[state] = self.weight_list[int(i/2)]
            if "bias" in state:
                states[state] = self.bias_list[int((i-1)/2)]
        self.load_state_dict(states)
    