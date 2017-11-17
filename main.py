#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.network import *
from common.trainer import *

network = Network()
optimizer = SGD(0.01)

trainer = Trainer()

trainer.startTraining()