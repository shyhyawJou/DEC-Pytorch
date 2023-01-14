import argparse
import os
from pathlib import Path as p
from time import time
import numpy as np
import random

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (load_data, set_data_plot, plot,
                   AutoEncoder, DEC,
                   pretrain, train, get_initial_center, 
                   accuracy)



def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=256, type=int, help='batch size')
    arg.add_argument('-pre_epoch', type=int, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', type=int, help='epochs for train DEC')
    arg.add_argument('-k', type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-worker', default=4, type=int, help='num of workers')
    arg.add_argument('-seed', type=int, default=None, help='torch random seed')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True) 
    else:
        for path in p(arg.save_dir).glob('*.png'):
            path.unlink()
        
    if arg.seed is not None:
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    tr_ds, test_ds = load_data(arg.bs, arg.worker)
         
    print('\ntrain num:', len(tr_ds.dataset))
    print('test num:', len(test_ds.dataset))
    
    # for visualize
    set_data_plot(tr_ds, test_ds, device)
    
    # train autoencoder
    ae = AutoEncoder().to(device)    
    print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
    opt = AdamW(ae.parameters())
    t0 = time()
    pretrain(ae, opt, tr_ds, device, arg.pre_epoch, arg.save_dir)
    t1 = time()
    
    # initial center
    t2 = time()
    ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
    center = get_initial_center(ae, tr_ds, device, arg.k) 
    t3 = time()
    
    # train dec
    print('\nload the best encoder and build DEC ...')
    ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
    dec = DEC(ae.encoder, center, alpha=1).to(device)  
    print(f'DEC param: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M') 
    opt = SGD(dec.parameters(), 0.01, 0.9, nesterov=True)
    t4 = time()
    train(dec, opt, tr_ds, device, arg.epoch, arg.save_dir)
    t5 = time()
    
    print()
    print('*' * 50)
    print('load the best DEC ...')
    dec = torch.load(f'{arg.save_dir}/DEC.pt', device)
    print('Evaluate ...')
    acc = accuracy(dec, test_ds, device)
    print(f'test acc: {acc:.4f}')
    print('*' * 50)
    plot(dec, arg.save_dir, 'test')

    print(f'\ntrain AE time: {t1 - t0:.2f} s')
    print(f'get inititial time: {t3 - t2:.2f} s')
    print(f'train DEC time: {t5 - t4:.2f} s')

    
    
if __name__ == '__main__':
    main()
