#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from collection import OrderedDict
from layers import *


class NetWork:
    def __init__(self,input_size,output_size,hidden_size_list,enable_weight_decay,weight_decay_lambda=None):
        self.input_size = input_size #入力ノード数
        self.output_size = output_layer #出力ノード数
        self.hidden_size_list = hidden_size_list #隠れ層のノード数
        self.hidden_layer_num = len(hidden_size_list) #隠れ層の数
        self.enable_weight_decay = enable_weight_decay #荷重減衰を利用するか (bool)
        self.weight_decay_lambda = weight_decay_lambda #荷重減衰係数
        self.layers = OrderedDict() #出力層以外
        self.output_layer = None #出力層
        # 重みを初期化
        self.params = {}
        self.initWeight(self.option)
        # 層を初期化
        self.initLayers()

    # 層の生成メソッド
    def initLayers(self):
        for index in range(self.hidden_layer_num):
            self.layers['Affine{}'.format(index)] = Affine(self.params['W{}'.format(index)],self.params['b{}'.format(index)])
            if activation_method == 'ReLU':
                self.layers['Activator{}'.format(index)] = ReLU()
            elif activation_method == 'sigmoid':
                self.layers['Activator{}'.format(index)] = Sigmoid()
        index = self.hidden_layer_num
        self.layers['Affine{}'.format(index)] = Affine(self.params['W{}'.format(index)],self.params['b{}'.format(index)])
        ## 出力層の生成
        self.output_layer = SoftmaxWithCrossEntropyError()

    # 重み初期化メソッド  *後で実装
    def initWeight(self):
        for index in range(self.hidden_layer_num):
            self.params['W{}'.format(index)] = np.random.randn() # 後で実装
            self.params['b{}'.format(index)] = np.zeros() # 後で実装

    # フォワード処理 (出力層の値を出す)
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 誤差関数の出力結果を返す
    # x: 入力データ
    # t: 正解ラベル
    def calcError(self,x,t):
        y = self.predict(x)
        if self.enable_weight_decay == True:
            weight_decay = 0
            ##############################
            #   ここに荷重減衰の実装をする
            ##############################
            return self.output_layer.forward(y,t) + weight_decay
        else:
            return self.output_layer.forward(y,t)

    # 勾配を計算
    def calcGradient(self, x, t):
        ## forward処理
        self.calcError(x,t)

        ## backward処理
        backward_input = 1
        backward_input = self.output_layer.backward(backward_input)

        layers = list(self.layers.values())
        layers = layers.reverse() #self.layersの順番を逆転させたリストを作成
        for layer in layers:
            backward_input = layer.backward(backward_input)

        # set return-values
        grads = {}
        for index in range(self.hidden_layer_num+1):
            if self.enable_weight_decay == True:
                grads['W{}'.format(index)] = self.layers['Affine{}'.format(index)].dW + self.weight_decay_lambda * self.layers['Affine{}'.format(index)].W
            else:
                grads['W{}'.format(index)] = self.layers['Affine{}'.format(index)].dW
            grads['b{}'.format(index)] = self.layers['Affine{}'.format(index)].db
        return grads

    # 正答率計算
    def calcAccuracy(self,x,t):
        y = self.predict(x) # (batch_size, class_num)
        y = np.argmax(y,axis=1) # 各出力ベクトルごとに尤もらしいインデックスを取得 (batch_size,)

        count = np.count_nonzero(y == t)
        accuracy = count / float(x.shape[0])

        return accuracy




















