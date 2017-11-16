#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np

# 汎用最適化手法クラス
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self):
        return 0

# 確率的勾配降下法
class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, params, grads):
        for key in params.keys():
            params[key] = params[key] - self.learning_rate * grads[key]