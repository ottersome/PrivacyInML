#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

DIVIDE = 25


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    #num_items = int(len(dataset) / num_users)
    num_items = int(len(dataset) / num_users / DIVIDE)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, bias):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {}
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))       # num_shards=6, num_imgs=10000: divide 60000 total data to (num_users * 2) folds
    idx_shard = [i for i in range(num_shards)]                                      # idx_shard=[0, 1, 2, 3, 4, 5]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}         # dict_users = {0: array([], dtype=int64), 1: array([], dtype=int64), 2: array([], dtype=int64)}
    idxs = np.arange(num_shards * num_imgs)                                         # idxs = [0 to 59999]
    labels = dataset.train_labels.numpy()                                           # each data's label has 6000 data

    # sort labels
    idxs_labels = np.vstack((idxs, labels))                                         # shape=(2. 6000)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]                       # sort the dataset according to labels 
    idxs = idxs_labels[0, :]                                                        # index of each data from label 0 to 9, idx[0:5999] is label0, idx[6000:11999] is label1 ...

    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))               # rand_set 2 out of 6
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)        # each user will have 2*10000 data, e.g. rand=1, 10000~20000, label 0~1~2


    # divide and assign  https://github.com/nicolagulmini/federated_learning/blob/master/dataset_split.py
    each_data_amount = int(np.floor(len(dataset.data)/num_users))  # each user has the same amount of data
    freq_amount = int(each_data_amount*bias)    # images of one particular class
    if freq_amount > 6000:                      # each class, at most 6000 images
        freq_amount = 6000
    uni_amount = each_data_amount-freq_amount   # how many uniformly distributed images

    # TODO: num_users too small, freq_amount=6000 (take the whole class), rest uniform select; num_users>10 work 

    for i in range(num_users):
        #freq_class = np.random.randint(0, 9)    # each user has a most frequent class
        freq_class = i
        dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs[freq_class*6000: (freq_class+1)*6000] , freq_amount, replace=False)), axis=0)
        dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs , uni_amount, replace=False)), axis=0)
        np.random.shuffle(dict_users[i])

    return dict_users

def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users



