#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np


# 活性化関数
def ReLU(x):
    return np.maximum(0,x)

def grad_ReLU(x):
    ret = np.zeros_like(x)
    ret[x >= 0] = 1.0
    return ret

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def grad_sigmoid(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    ret = np.exp(x-np.max(x))
    return ret / np.sum(ret)


# 誤差評価関数
## 平均二乗誤差 mean squared error
def mse(y,t):
    ret = y - t
    ret = np.square(ret)
    return 0.5 * np.sum(ret)

# 交差エントロピー誤差 cross entropy error
# t: 正解ラベルのnumpy配列 (N,)
# y: 出力 numpy配列 (N,n)
def cee(y,t):
    batch_size = y.shape[0]
    ret = np.log(y[np.arange(batch_size),t])
    return -np.sum(ret)/batch_size
