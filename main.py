#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.network import *
from common.trainer import *
from common.optimizer import *


net = Network(input_size=784,output_size=10,hidden_layer_size_list=[100,100],activator='ReLU',enable_weight_decay=False,weight_decay_lambda=None)
opt = SGD(learning_rate=0.01)

trainer = Trainer(network=net, optimizer=opt, x_train, t_train, x_test, t_test, dataset_loops=20, batch_size=100)

trainer.startTraining()

trainer.showAccuracy()