from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np

import torch



def accuracy(model, ds, device):
    truth, pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in ds:
            x = x.to(device)
            truth.append(y)
            pred.append(model(x).max(1)[1].cpu())        
    
    confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred).numpy())
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    
    return acc
