# GAN
## Generative adversarial nets，using pytorch，mnist dataset  
使用数据集mnist的训练集部分  
## 这是最简单的能读懂的GAN实现，代码具有非常高的可读性，全部加有注释  
### 准备,总共需要以下导入库  
`import os`  
`import matplotlib.pyplot as plt`  
`import itertools`  
`import pickle`  
`import imageio`  
`import torch`  
`import torch.nn as nn`  
`import torch.nn.functional as F`  
`import torch.optim as optim`  
`from torchvision import datasets, transforms`  
`from torch.autograd import Variable`  
### 使用方法  
平台是python3.x  
python main.py
### 说明  
* mnist数据集自动下载，选择download=True，也可进行变换，只是用到了它的训练集（剩下的测试集没有使用）
`datasets.MNIST('data', train=True, download=True, transform=transform)`  
*   代码里面有详细的讲解，应该比较容易弄懂

