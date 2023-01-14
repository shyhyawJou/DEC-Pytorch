from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, STL10



def load_data(batch_size, num_worker):
    tr_ds = MNIST('mnist', True, None, download=True)
    tr_ds = list(zip(tr_ds.data.flatten(1) / 255., tr_ds.targets))
    test_ds = MNIST('mnist', False, None, download=True)
    test_ds = list(zip(test_ds.data.flatten(1) / 255., test_ds.targets))
    
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
    
    