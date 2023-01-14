from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.nn import Parameter

from .nn import get_p
from .for_eval import accuracy



DATA_PLOT = None


def set_data_plot(tr_ds, test_ds, device):
    global DATA_PLOT

    # select 100 sample per class  
    tr_x, tr_y = [], []
    count = torch.zeros(10, dtype=torch.int)
    for batch in tr_ds:
        for data, label in zip(*batch):
            if count[label] < 100:
                tr_x.append(data[None])
                tr_y.append(label[None])
            count[label] += 1
    tr_x, tr_y = torch.cat(tr_x).to(device), torch.cat(tr_y).to(device)

    # select 100 sample per class  
    test_x, test_y = [], []
    count = torch.zeros(10, dtype=torch.int)
    for batch in test_ds:
        for data, label in zip(*batch):
            if count[label] < 100:
                test_x.append(data[None])
                test_y.append(label[None])
            count[label] += 1
    test_x, test_y = torch.cat(test_x).to(device), torch.cat(test_y).to(device)

    DATA_PLOT = {'train': (tr_x, tr_y), 
                 'test': (test_x, test_y)}


def get_initial_center(model, ds, device, n_cluster):
    # fit
    print('\nbegin fit kmeans++ to get initial cluster centroids ...')
    
    model.eval()
    with torch.no_grad():
        feature = []
        for x, _ in ds:
            x = x.to(device)
            feature.append(model.encoder(x).cpu())
            
    kmeans = KMeans(n_cluster).fit(torch.cat(feature).numpy())
    center = Parameter(torch.tensor(kmeans.cluster_centers_, 
                                    device=device, 
                                    dtype=torch.float))
    
    return center


def pretrain(model, opt, ds, device, epochs, save_dir):     
    print('begin train AutoEncoder ...')
    
    loss_fn = nn.MSELoss()
    n_sample, n_batch = len(ds.dataset), len(ds)
    model.train() 
    loss_h = History('min')
    
    # fine-tune
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        loss = 0.
        for i, (x, y) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)
            _, gen = model(x)
            batch_loss = loss_fn(x, gen)
            batch_loss.backward()
            opt.step()
            loss += batch_loss * y.numel()
            print(f'{i}/{n_batch}', end='\r')

        loss /= n_sample
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/fine_tune_AE.pt')
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')
                   
        
def train(model, opt, ds, device, epochs, save_dir):
    print('begin train DEC ...')
    
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    n_sample, n_batch = len(ds.dataset), len(ds)
    loss_h, acc_h = History('min'), History('max')
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        model.train()
        loss = 0.
        for i, (x, y) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)            
            q = model(x)
            batch_loss = loss_fn(q.log(), get_p(q))
            batch_loss.backward()
            opt.step()
            loss += batch_loss * y.numel()
            print(f'{i}/{n_batch}', end='\r')
            
        loss /= n_sample
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/DEC.pt')
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')

        acc = accuracy(model, ds, device)
        acc_h.add(acc)
        print(f'acc : {acc:.4f}  max acc : {acc_h.best:.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')
        
        if epoch % 5 == 0:
            plot(model, save_dir, 'train', epoch)
                   
    df = pd.DataFrame(zip(range(1, epoch+1), loss_h.history, acc_h.history), columns=['epoch', 'loss', 'acc'])
    df.to_excel(f'{save_dir}/train.xlsx', index=False)
    
               
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value.item())
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')
    

def plot(model, save_dir, target='train', epoch=None):
    assert target in {'train', 'test'}
    print('plotting ...')
    
    model.eval()
    with torch.no_grad():
        feature = model.encoder(DATA_PLOT[target][0])
        pred = model.cluster(feature).max(1)[1].cpu().numpy()
        
    feature_2D = TSNE(2).fit_transform(feature.cpu().numpy())
    plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, pred, cmap='Paired')
    if epoch is None:
        plt.title(f'Test data')
        plt.savefig(f'{save_dir}/test.png')
    else:
        plt.title(f'Epoch: {epoch}')
        plt.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt.close()