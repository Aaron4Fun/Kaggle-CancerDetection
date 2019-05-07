from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = M.resnet50(pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, 1),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def unfreeze_model(self):
        for param in self.net.parameters():
            param.requires_grad = True
            
    def freeze_model(self):
        for param in self.net.parameters():
            param.requires_grad = False        
        for param in self.net.fc.parameters():
            param.requires_grad = True        

    def forward(self, x):
        return self.net(x)


class Densenet169(nn.Module):
    def __init__(self, pretrained=True):
        super(Densenet169, self).__init__()
        self.model = M.densenet169(pretrained=pretrained)
        self.linear = nn.Linear(1000+2, 16)
        self.bn = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.out = nn.Linear(16, 1)   
        
    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = True
        for param in self.bn.parameters():
            param.requires_grad = True
        for param in self.out.parameters():
            param.requires_grad = True 


    def freeze_basemodel(self):
        for param in self.model.parameters():
            param.requires_grad = False     
        for param in self.linear.parameters():
            param.requires_grad = True
        for param in self.bn.parameters():
            param.requires_grad = True
        for param in self.out.parameters():
            param.requires_grad = True 
            
            
    def forward(self, x):
        out = self.model(x)
        batch = out.shape[0]
        max_pool, _ = torch.max(out, 1, keepdim=True)
        avg_pool = torch.mean(out, 1, keepdim=True)

        out = out.view(batch, -1)
        conc = torch.cat((out, max_pool, avg_pool), 1)

        conc = self.linear(conc)
        conc = self.elu(conc)
        conc_feat = self.bn(conc)
        conc = self.dropout(conc_feat)

        res = self.out(conc)

        return conc_feat, res
    
densenet169 = partial(Densenet169)
resnet50 = partial(ResNet50)