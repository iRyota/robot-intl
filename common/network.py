#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np
from layers import *


class NetWork:
    def __init__(self,input_size,output_size,hidden_size_list,enable_weight_decay,weight_decay_lambda=None):
        self.input_size = input_size #入力ノード数
        self.output_size = output_layer #出力ノード数
        self.hidden_size_list = hidden_size_list #隠れ層のノード数
        self.hidden_layer_num = len(hidden_size_list) #隠れ層の数
        self.enable_weight_decay = enable_weight_decay #荷重減衰を利用するか (bool)
        self.weight_decay_lambda = weight_decay_lambda #荷重減衰係数
        self.layers = [] #出力層以外
        self.output_layer = [] #出力層
        # 重みを初期化
        self.params = {}
        self.initWeight(self.option)
        # 層を初期化
        self.initLayers()

    # 層の生成メソッド
    def initLayers(self):
        for index in range(self.hidden_layer_num):
            self.layers.append(Affine(self.params['W'+str(index)],self.params['b'+str(index)]))
            if activation_method == 'ReLU':
                self.layers.append(ReLU())
            elif activation_method == 'sigmoid':
                self.layers.append(Sigmoid())
        index = self.hidden_layer_num
        self.layers.append(Affine(self.params['W'+str(index),self.params['b'+str(index)]]))
        ## 出力層の生成
        self.output_layer.append(SoftmaxAndError())

    # 重み初期化メソッド  *後で実装
    def initWeight(self):
        ret = []
        return ret

    # フォワード処理 (出力層の値を出す)
    def predict(self, x):
        for layer in self.layers:
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

    # 正答率計算
    def calcAccuracy(self,x,t):
        # 後で実装
        accuracy = 0
        return accuracy

    # Back Propergation (誤差逆伝播) 用に勾配を計算
    def calcBPGradient(self, x, t):
        # 後で実装
        ## forward処理

        ## backward処理

        # set return-values
        grads = {}
        for index in range(self.hidden_layer_num+1):
            grads['W'+str(index)] = self.layers[2*index].dW
            if self.enable_weight_decay == True:
                grads['W'+str(index)] = self.layers[2*index].dW + self.weight_decay_lambda * self.layers[2*index].W
            grads['b'+str(index)] = self.layers[2*index].db
        return grads





















