import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T



def load_data(batch_size, num_worker):
    t = T.Compose([T.ToTensor(),
                   nn.Flatten(0)])
    
    tr_ds = MNIST('mnist', True, t, download=True)
    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    test_ds = MNIST('mnist', False, t, download=True)
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]
    
    tr_ds = DataLoader(tr_ds, 
                       batch_size, 
                       shuffle=True, 
                       num_workers=num_worker, 
                       pin_memory=True)
   
    test_ds = DataLoader(test_ds, 
                         batch_size,
                         shuffle=False,
                         num_workers=num_worker, 
                         pin_memory=True)
    
    return tr_ds, test_ds
