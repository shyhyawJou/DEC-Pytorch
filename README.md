# Overview
original paper -> [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335)

# Clustering Result
- __MNIST__  
Accuracy of training data -> 87.57%  
Accuracy of test data     -> 87.53%
<table>
  <tr>
    <td><img src="assets/loss.png" alt="image1"></td>
    <td><img src="assets/acc.png" alt="image2"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="assets/epoch_5.png" alt="image1"></td>
    <td><img src="assets/epoch_30.png" alt="image1"></td>
  </tr>
</table>

# Usage
```
python train.py -bs 256 -k 10 -pre_epoch 30 -epoch 30 -seed 1000
```
