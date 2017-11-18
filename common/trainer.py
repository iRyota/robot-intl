#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

from common.optimizer import *


# 学習用クラス
class Trainer:
    def __init__(self, network, optimizer, x_train, t_train, x_test, t_test, dataset_loops, batch_size):
        self.network = network      # Networkオブジェクト
        self.optimizer = optimizer  # Optimizerオブジェクト
        self.x_train = x_train      # 訓練データ
        self.t_train = t_train      # 教師データ
        self.x_test = x_test        # テスト入力データ
        self.t_test = t_test        # テスト正解データ
        self.train_accuracy_list = []
        self.test_accuracy_list = []

        self.batch_size = batch_size
        self.train_size = x_train.shape[0]

        self.dataset_loops = dataset_loops # データセットを何周するか
        self.iterators_per_loop = max(int(self.train_size/batch_size),1)
        self.total_loops = int(dataset_loops * self.iterators_per_loop) # バッチ学習を計何回するか
        self.current_iterator = 0
        self.current_loop_index = 0

    # 学習（1ループ分）
    def train(self):
        batch_index = np.random.choice(self.train_size,self.batch_size)
        x_batch = self.x_train[batch_index]
        t_batch = self.t_train[batch_index]

        # 誤差逆伝播法により勾配を求める
        grads = self.network.calcGradient(x_batch, t_batch)

        # 重みデータの更新
        self.network.params = self.optimizer.update(self.network.params, grads)
        self.network.updateLayers()

        # 正答率計算・格納
        if self.current_iterator == 0:
            train_accuracy = self.network.calcAccuracy(self.x_train,self.t_train)
            test_accuracy = self.network.calcAccuracy(self.x_test,self.t_test)
            self.train_accuracy_list.append(train_accuracy)
            self.test_accuracy_list.append(test_accuracy)


    # 学習開始関数 (これが呼び出される)
    def startTraining(self):
        for i in range(self.total_loops):
            self.train()
            self.current_iterator += 1
            if self.current_iterator >= self.iterators_per_loop:
                self.current_iterator = 0
                self.current_loop_index += 1
                print('loop: {}/{} done'.format(self.current_loop_index,self.dataset_loops))
        print('training done')

    # 正答率の可視化
    def showAccuracy(self):
        plt.plot(np.arange(self.dataset_loops+1)[1:],100*np.array(self.train_accuracy_list),label='Train Data')
        plt.plot(np.arange(self.dataset_loops+1)[1:],100*np.array(self.test_accuracy_list),label='Test Data')
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy (%)')
        plt.ylim([0,100])
        plt.show()
        plt.clf()








