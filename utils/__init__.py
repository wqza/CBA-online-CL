# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(state, model_name, save_path):
    torch.save(state, os.path.join(save_path, model_name))


def plot_accuracy(save_path, name, test_acc, train_acc=None):
    plt.figure(dpi=200)
    plt.plot(np.arange(len(test_acc))+1, test_acc, c='darkorange')
    if train_acc is not None:
        plt.plot(np.arange(len(train_acc))+1, train_acc, c='dodgerblue')
        plt.legend(['test_acc', 'train_acc'])
    else:
        plt.legend(['test_acc'])
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.close()


def get_feature(dataloader, model, sampling=None):
    status = model.net.training
    model.net.eval()
    all_features, all_labels = [], []
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if len(data) == 2:
                inputs, labels = data
            else:
                inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            features = model.net(inputs, returnt='feature')
            all_features.append(features)
            all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    if sampling is not None:
        rand_idx = np.random.choice(np.arange(len(all_features)), sampling, replace=False)
        all_features = all_features[rand_idx]
        all_labels = all_labels[rand_idx]

    model.net.train(status)
    return all_features, all_labels


def get_all_feature_for_tsne(model, train_loader, test_loaders, sampling=None, transform=None):
    # current training data
    train_feature, train_label = get_feature(train_loader, model, sampling)

    # current buffer sample
    buffer_data = model.buffer.get_all_data(transform)
    batch_size = 128
    buffer_size = model.buffer.buffer_size
    iter_num = buffer_size // batch_size + 1 if buffer_size % batch_size > 0 else buffer_size // batch_size
    buffer_loader = [(buffer_data[0][i*batch_size: (i+1)*batch_size],
                      buffer_data[1][i*batch_size: (i+1)*batch_size]) for i in range(iter_num)]
    buffer_feature, buffer_label = get_feature(buffer_loader, model, min(sampling, buffer_size))

    # test data
    test_features, test_labels = [], []
    for k, test_loader in enumerate(test_loaders):
        test_feature, test_label = get_feature(test_loader, model, sampling)
        test_features.append(test_feature)
        test_labels.append(test_label)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    total_feature = np.array(torch.cat((train_feature, buffer_feature, test_features)).detach().cpu())
    total_label = np.array(torch.cat((train_label, buffer_label, test_labels)).detach().cpu())
    total_train_or_test = torch.cat((torch.ones_like(train_label) * 0,   # 0 means current training set
                                     torch.ones_like(buffer_label) * 1,  # 1 means buffer set
                                     torch.ones_like(test_labels) * 2))  # 2 means test set for all seen classes
    total_train_or_test = np.array(total_train_or_test.detach().cpu())
    return total_feature, total_label, total_train_or_test


def two_sets_distance(set1, set2, distance='all'):
    """
    Input:
        set1: [sample_num, vector_dim] the 1st vector set
        set2: [sample_num, vector_dim] the 2nd vector set
        distance: should be one of the 'all', 'cosine', 'l2', 'hausdorff'
    Output:
    """
    mu1 = np.mean(set1, axis=0)
    mu2 = np.mean(set2, axis=0)

    dist_list = []
    if distance == 'cosine' or 'all':
        dist = 1 - np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2))
        dist_list.append(dist)
    if distance == 'l2' or 'all':
        dist = np.sqrt(np.sum((mu1 - mu2) ** 2))
        dist_list.append(dist)
    if distance == 'hausdorff' or 'all':
        min_dd = []
        for i in range(set1.shape[0]):
            dd = np.sum((set1[i] - set2) ** 2, axis=1)
            min_dd.append(dd.min())
        min_dd = np.array(min_dd)
        dist = np.sqrt(min_dd.max())
        dist_list.append(dist)

    if distance == 'all':
        return dist_list
    else:
        return dist_list[0]


def calculate_distance(set1, set1_label, set2, set2_label, distance='all'):
    """
    Calculate the distance between two sets for all seen classes.
    e.g. calculate the overfit between the feature of the train set and test set
    e.g. calculate the drift between the feature of the old model and current model on test set.
    """
    cls_set = np.intersect1d(np.unique(set1_label), np.unique(set2_label))
    cls_dist = []
    for cls in cls_set:
        cls = int(cls)
        cls_dist.append(two_sets_distance(set1[set1_label == cls], set2[set2_label == cls], distance=distance))
    cls_dist = np.array(cls_dist)
    if distance == 'all':
        return {'cosine':    cls_dist[:, 0],
                'l2':        cls_dist[:, 1],
                'hausdorff': cls_dist[:, 2]}
    else:
        return cls_dist




