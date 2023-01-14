import torch
from torch import nn



class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Encoder(784, 500, True),
                                     Encoder(500, 500, True),
                                     Encoder(500, 2000, True),
                                     Encoder(2000, 10, False))
        self.decoder = nn.Sequential(Decoder(10, 2000, True),
                                     Decoder(2000, 500, True),
                                     Decoder(500, 500, True),
                                     Decoder(500, 784, False))
            
    def forward(self, x):
        x  = self.encoder(x)
        gen = self.decoder(x)
        return x, gen


class Cluster(nn.Module):
    def __init__(self, center, alpha):
        super().__init__()
        self.center = center
        self.alpha = alpha

    def forward(self, x):
        square_dist = torch.pow(x[:, None, :] - self.center, 2).sum(dim=2)
        nom = torch.pow(1 + square_dist / self.alpha, -(self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        return nom / denom


def get_p(q):
    with torch.no_grad():
        f = q.sum(dim=0, keepdim=True)
        nom = q ** 2 / f
        denom = nom.sum(dim=1, keepdim=True)
    return nom / denom
    
    
class DEC(nn.Module):
    def __init__(self, encoder, center, alpha=1):
        super().__init__()
        self.encoder = encoder
        self.cluster = Cluster(center, alpha)

    def forward(self, x):
        x = self.encoder(x)
        x = self.cluster(x)
        return x
