#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def grad(f,x):
    delta = 1.0e-4
    ret = np.zeros_like(x)

    iterator = np.nditer(x, flags=['multi_index'])
    while not iterator.finished:
        index = iterator.multi_index
        tmp = x[index]

        x[index] = tmp + delta
        fx_1 = f(x)
        x[index] = tmp - delta
        fx_2 = f(x)

        ret[index] = (fx_1-fx_2)/(2*delta)
        x[index] = tmp
        iterator.iternext()

    return ret