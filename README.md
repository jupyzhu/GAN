# GAN
## Generative adversarial nets，using pytorch，mnist dataset  
使用数据集mnist的训练集部分  
## 这是最简单的GAN实现  
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
python GAN.py
### 说明  
* mnist数据集自动下载，选择download=True，也可进行变换，只是用到了它的训练集（剩下的测试集没有使用）
`datasets.MNIST('data', train=True, download=True, transform=transform)`  
*   会在代码所在文件目录下面创建GAN_results的文件夹，该文件夹包含了6个文件：生成器的参数模型，判别器的参数模型，两个loss的训练历史值，
