#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np


class Layer:
    def __init__(self, weight, bias):
        self.weight = weight  # np.array
        self.bias = bias
        # self.params = [weight; bias]