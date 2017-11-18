#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import *


class NetWork:
    def __init__(self,input_size,output_size,hidden_layer_size_list,activator,enable_weight_decay,weight_decay_lambda=None):
        self.input_size = input_size #入力ノード数
        self.output_size = output_size #出力ノード数
        self.hidden_layer_size_list = hidden_layer_size_list #隠れ層のノード数のリスト
        self.hidden_layer_num = len(hidden_layer_size_list) #隠れ層の数
        self.activator = activator #活性化関数
        self.enable_weight_decay = enable_weight_decay #荷重減衰を利用するか (bool)
        self.weight_decay_lambda = weight_decay_lambda #荷重減衰係数
        self.layers = OrderedDict() #出力層以外
        self.output_layer = None #出力層
        # 重みを初期化
        self.params = {}
        self.initWeight()
        # 層を初期化
        self.initLayers()

    # 層の生成メソッド
    def initLayers(self):
        for index in range(self.hidden_layer_num):
            self.layers['Affine{}'.format(index)] = Affine(self.params['W{}'.format(index)],self.params['b{}'.format(index)])
            if self.activator == 'ReLU':
                self.layers['Activator{}'.format(index)] = ReLU()
            elif self.activator == 'sigmoid':
                self.layers['Activator{}'.format(index)] = Sigmoid()
        index = self.hidden_layer_num
        self.layers['Affine{}'.format(index)] = Affine(self.params['W{}'.format(index)],self.params['b{}'.format(index)])
        ## 出力層の生成
        self.output_layer = SoftmaxWithCrossEntropyError()

    def updateLayers(self):
        for index in range(self.hidden_layer_num+1):
            self.layers['Affine{}'.format(index)].W = self.params['W{}'.format(index)]
            self.layers['Affine{}'.format(index)].b = self.params['b{}'.format(index)]

    # 重み初期化メソッド
    def initWeight(self):
        # 各層のノードの数の情報が必要
        layer_size_list = [self.input_size] + self.hidden_layer_size_list + [self.output_size]

        for index in range(len(layer_size_list)-1):
            if self.activator=='ReLU':
                self.params['W{}'.format(index)] = np.random.randn(layer_size_list[index],layer_size_list[index+1]) / np.sqrt(layer_size_list[index]/2.0) # Heの初期値
            elif self.activator=='sigmoid':
                self.params['W{}'.format(index)] = np.random.randn(layer_size_list[index],layer_size_list[index+1]) / np.sqrt(layer_size_list[index]) # Xavierの初期値
            self.params['b{}'.format(index)] = np.zeros(layer_size_list[index+1])

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
        layers.reverse() #self.layersの順番を逆転させたリストを作成
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




















