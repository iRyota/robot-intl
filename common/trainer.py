#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np

from Optimizer.sgd import *


# 学習用クラス
class Trainer:
    def __init__(self, network, optimizer, x_train, t_train, x_test, t_test):
        self.network = network      # Networkオブジェクト
        self.optimizer = optimizer  # Optimizerオブジェクト
        self.x_train = x_train      # 訓練データ
        self.t_train = t_train      # 教師データ
        self.x_test = x_test        # テスト入力データ
        self.t_test = t_test        # テスト正解データ
        self.accuracy_array = []

    # 学習（1ループ分）
    def train(self):

        # 誤差逆伝播法により勾配を求める
        grads = self.network.gradient(x_batch, t_batch)

        # 重みデータの更新
        self.optimizer.update(self.network.params, grads)



        # # 正答率計算・格納
        # accuracy = self.calcAccuracy()
        # self.accuracy_array.append(accuracy)

    # 学習開始関数 (これが呼び出される)
    def startTraining(self):
        for i in range(self.loop_num):
            self.train()
