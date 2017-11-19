#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
import gzip


class Dataset:
    def __init__(self,dataset_dir,save_as,image_size,train_size,test_size,train_image_file,train_label_file,test_image_file,test_label_file):
        self.dataset_dir = dataset_dir
        self.save_as = "{}/{}.pkl".format(dataset_dir,save_as)
        self.image_size = image_size
        self.train_size = train_size
        self.test_size = test_size
        self.train_image_file = train_image_file # .gzファイルのファイル名
        self.train_label_file = train_label_file # .gzファイルのファイル名
        self.test_image_file = test_image_file # .gzファイルのファイル名
        self.test_label_file = test_label_file # .gzファイルのファイル名

        self.dataset = {}

    def loadImageData(self,key_name,file_name):
        file_path = "{}/{}".format(self.dataset_dir,file_name)
        print("Converting {} to np.array".format(file_name))
        with gzip.open(file_path, 'rb') as f:
            self.dataset[key_name] = np.frombuffer(f.read(), np.uint8, offset=16)
        self.dataset[key_name] = self.dataset[key_name].reshape(-1, self.image_size)
        print("Done")

    def loadLabelData(self,key_name,file_name):
        file_path = "{}/{}".format(self.dataset_dir,file_name)
        print("Converting {} to np.array".format(file_name))
        with gzip.open(file_path, 'rb') as f:
            self.dataset[key_name] = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")

    def saveDataset(self):
        self.loadImageData('train_image',self.train_image_file)
        self.loadLabelData('train_label',self.train_label_file)
        self.loadImageData('test_image',self.test_image_file)
        self.loadLabelData('test_label',self.test_label_file)

        print("Creating pickle file")
        with open(self.save_as, 'wb') as f:
            pickle.dump(self.dataset,f,-1)
        print("Done")

    def loadDataset(self):
        with open(self.save_as, 'rb') as f:
            self.dataset = pickle.load(f)

        # 正規化
        for key in ('train_image', 'test_image'):
            self.dataset[key] = self.dataset[key].astype(np.float32)
            self.dataset[key] /= 255.0

    # 各画素に(noise_ratio)%の確率でノイズがかかる
    def putNoise(self,noise_ratio):
        rand = np.random.rand(*self.dataset['train_image'].shape)
        mask = np.random.rand(*self.dataset['train_image'].shape) > noise_ratio
        self.dataset['train_image'] = self.dataset['train_image'] * mask + 1*np.logical_not(mask)