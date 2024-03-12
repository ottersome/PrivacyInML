#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib

matplotlib.use("Agg")
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from models.Fed import FedWeightAvg
from models.Nets import MLP, CharLSTM, CNNCifar, CNNFemnist, CNNMnist
from models.test import test_img
from models.Update import LocalUpdate
from utils.dataset import FEMNIST, ShakeSpeare
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_noniid, mnist_iid, mnist_noniid
from utils.utility import quantization, top_k_sparsificate_model_weights

if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    # parse args
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    # # # manually set input arguments
    # args.dataset = 'fashion-mnist'
    # args.num_channels = 1
    # args.num_users = 3
    # args.frac = 1
    # args.model = 'cnn'
    # args.epochs = 10

    print(
        args.compression_type
        + "_Rate"
        + str(args.R)
        + "_Mvalue"
        + str(args.M)
        + "_%"
        + str(args.sp_perc)
    )

    # compression_type = 'GenNorm'
    # QUANTIZATION_M = 0

    # load dataset and split users
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset_train = datasets.MNIST(
            "./data/mnist/", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            "./data/mnist/", train=False, download=True, transform=trans_mnist
        )
        # TODO
        # create smaller dataset for 2-user or max 5-user

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.bias)
    elif args.dataset == "cifar":
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trans_cifar_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset_train = datasets.CIFAR10(
            "./data/cifar", train=True, download=True, transform=trans_cifar_train
        )
        dataset_test = datasets.CIFAR10(
            "./data/cifar", train=False, download=True, transform=trans_cifar_test
        )
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == "fashion-mnist":
        trans_fashion_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset_train = datasets.FashionMNIST(
            "./data/fashion-mnist",
            train=True,
            download=True,
            transform=trans_fashion_mnist,
        )
        dataset_test = datasets.FashionMNIST(
            "./data/fashion-mnist",
            train=False,
            download=True,
            transform=trans_fashion_mnist,
        )
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.bias)
    elif args.dataset == "femnist":
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit("Error: femnist dataset is naturally non-iid")
        else:
            print(
                "Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid"
            )
    elif args.dataset == "shakespeare":
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit("Error: ShakeSpeare dataset is naturally non-iid")
        else:
            print(
                "Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid"
            )
    else:
        exit("Error: unrecognized dataset")
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == "cnn" and args.dataset == "cifar":
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == "cnn" and (
        args.dataset == "mnist" or args.dataset == "fashion-mnist"
    ):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == "femnist" and args.model == "cnn":
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == "shakespeare" and args.model == "lstm":
        net_glob = CharLSTM().to(args.device)
    elif args.model == "mlp":
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(
            args.device
        )
    else:
        exit("Error: unrecognized model")

    print(net_glob)  # TODO: correlation MLP vs. CNN
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    acc_test = []
    clients = [
        LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        for idx in range(args.num_users)
    ]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(
        args.num_users
    )
    for iter in range(args.epochs):
        w_locals, loss_locals, weight_locols = [], [], []

        # idxs_users = np.random.choice(clients_index_array, m, replace=False)
        idxs_users = np.sort(np.random.choice(clients_index_array, m, replace=False))

        for idx in idxs_users:
            w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))

            # layer_name = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
            # for layer in layer_name:
            #     gradient = w[layer] - w_glob[layer]

            #     # sparsification
            #     #print('sparse level:', args.sp_perc/(100))
            #     sparse_gradient = top_k_sparsificate_model_weights(gradient, args.sp_perc/(100))

            #     # quantization
            #     quantized_gradient = quantization(sparse_gradient, args)

            #     w[layer] = quantized_gradient + w_glob[layer]

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

        acc_test.append(acc_t.item())

    rootpath = "./log"
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(
        rootpath
        + "/accfile_fed_{}_bias{}_{}_{}_iid{}_{}_R{}_M{}_sp%{}.dat".format(
            args.dataset,
            args.bias,
            args.model,
            args.epochs,
            args.iid,
            args.compression_type,
            args.R,
            args.M,
            args.sp_perc,
        ),
        "w",
    )

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write("\n")
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel("test accuracy")
    plt.savefig(
        rootpath
        + "/fed_{}_{}_{}_C{}_iid{}_acc.png".format(
            args.dataset, args.model, args.epochs, args.frac, args.iid
        )
    )
