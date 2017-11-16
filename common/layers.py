#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from functions import *

class ReLU:
    def __init__(self):
        self.filter = None # 0以上だったらTrue, 0未満だったらFalseが入っているフィルタ

    def forward(self, x):
        self.filter = (x < 0)
        forward_output = x.copy()
        forward_output[self.filter] = 0
        return forward_output

    def backward(self, backward_input):
        back[self.filter] = 0
        backward_output = backward_input
        return backward_output


class Sigmoid:
    def __init__(self):
        self.forward_output = None # 出力

    def forward(self,x):
        forward_output = sigmoid(x)
        self.forward_output = forward_output
        return forward_output

    def backward(self,backward_input):
        backward_output = backward_input * self.forward_output * (1.0 - self.forward_output)
        return backward_output

class Affine:
    def __init__(self, weight, bias):
        self.W = weight # 重み
        self.b = bias # バイアス
        self.dW = None # 重みの微分
        self.db = None # バイアスの微分

        self.forward_input = None
        self.forward_input_shape = None

    def forward(self,x):
        self.forward_input_shape = x.shape
        x = x.reshape(x.shape[0],1)
        self.forward_input = x

        forward_output = np.dot(self.forward_input,self.W) + self.b
        return forward_output

    def backward(self,backward_input):
        backward_output = np.dot(backward_input,self.W.T)
        self.dW = np.dot(self.forward_input.T,backward_input)
        self.db = np.sum(backward_input,axis=0)
        return backward_output

class SoftmaxWithCrossEntropyError:
    def __init__(self):
        self.forward_output = None # 出力
        self.teacher = None # 教師ラベル
        self.error = None # 誤差

    def forward(self,x,t):
        self.teacher = t # np.array (batch_size,)
        self.forward_output = softmax(x)
        self.error = cee(self.forward_output,self.teacher)

        return self.error

    def backward(self,backward_input=1):
        batch_size = self.t.shape[0]
        backward_output = self.y.copy()
        backward_output[np.arange(batch_size),self.teacher] -= 1
        backward_output = backward_output/batch_size
        return backward_output

class SoftmaxWithMeanSquaredError:
    def __init__(self):
        self.forward_output = None
        self.teacher = None
        self.error = None

    def forward(self,x,t):
        self.teacher = t
        self.forward_output = softmax(x)
        self.error = mse(self.forward_output,self.teacher)

        return self.error

    def backward(self,backward_input=1):
        batch_size = self.t.shape[0]
        backward_output = (self.forward_output - self.teacher)/batch_size
        return backward_output





