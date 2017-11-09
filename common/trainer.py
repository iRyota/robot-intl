#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np


# 学習用クラス
class Trainer:
    def __init__(self, network, optimizer, x_train, t_train):
        self.network = network      # object
        self.optimizer = optimizer  # object
        self.x_train = x_train      # 訓練データ
        self.t_train = t_train      # 教師データ
        self.accuracy_array = []

    # 学習（1ループ分）
    def train(self):

        # 重みデータの更新
        grads = self.network.gradient()
        self.optimizer.update(self.network.params, grads)


        # 正答率計算・格納
        accuracy = self.calcAccuracy()
        self.accuracy_array.append(accuracy)

    # 学習開始関数 (これが呼び出される)
    def startTraining(self):
        for i in range(self.loop_num):
            self.train()
