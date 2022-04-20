import torch
import torch.nn as nn
from torch.quantization import QuantStub,DeQuantStub,fuse_modules
from models.networks.assisting_layers.ContainerLayers import ContainerLinear

class ConvertMLP(nn.Module):
    def __init__(self,model):
        super(ConvertMLP,self).__init__()
        modules=[]
        for name,module in model.named_modules():
            if isinstance(module,ContainerLinear):
                modules.append(nn.Linear(module.in_features,module.out_features))
            else:
                modules.append(module)
        modules = modules[2:]  
        self.layers=nn.Sequential(*modules)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x
    def fuse(self):
        pass
    
class ConvertMLP2(ConvertMLP):
    def __init__(self,model):
        super(ConvertMLP2,self).__init__(model)

    def fuse(self):
        fuse_modules(self, [['layers.0','layers.1']], inplace=True)

    
class ConvertMLP3(ConvertMLP):
    def __init__(self,model):
        super(ConvertMLP3,self).__init__(model)

    def fuse(self):
        fuse_modules(self, [['layers.0','layers.1'],['layers.2','layers.3']], inplace=True)
