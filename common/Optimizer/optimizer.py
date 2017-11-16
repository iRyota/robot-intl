#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np

# 汎用最適化手法クラス
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self):
        return 0