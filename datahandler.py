import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import random
import numpy as np


def get_class_i_indices(y, i):
    y = np.array(y)
    pos_i = np.argwhere(y == i)
    pos_i = list(pos_i[:, 0])
    random.shuffle(pos_i)

    return pos_i


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    random.shuffle(indices)
    return indices


def dict_indices(dataset):
    idx_classes = {}
    for i in range(10):
        idx_classes[i] = get_indices(dataset, i)
    return idx_classes


def get_indx_balanced_train_subset(dict_indices, k):
    # print(len(dict_indices[0]))
    indx_balanced_subset = []
    for i in range(10):
        p10_idx = len(dict_indices[i]) // 10
        # print(p10_idx)
        indx_balanced_subset += dict_indices[i][k:k + p10_idx]
    return indx_balanced_subset


def get_indx_balanced_test_subset(dict_indices, k):
    indx_balanced_subset = []
    for i in range(10):
        indx_balanced_subset += dict_indices[i][k:k + 100]
    return indx_balanced_subset


def get_subset_data(y_train, y_test):

    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                 'truck': 9}

    plane_indices = get_class_i_indices(y_train, classDict['plane'])
    car_indices = get_class_i_indices(y_train, classDict['car'])
    bird_indices = get_class_i_indices(y_train, classDict['bird'])
    cat_indices = get_class_i_indices(y_train, classDict['cat'])
    deer_indices = get_class_i_indices(y_train, classDict['deer'])
    dog_indices = get_class_i_indices(y_train, classDict['dog'])
    frog_indices = get_class_i_indices(y_train, classDict['frog'])
    horse_indices = get_class_i_indices(y_train, classDict['horse'])
    ship_indices = get_class_i_indices(y_train, classDict['ship'])
    truck_indices = get_class_i_indices(y_train, classDict['truck'])

    plane_indices_test = get_class_i_indices(y_test, classDict['plane'])
    car_indices_test = get_class_i_indices(y_test, classDict['car'])
    bird_indices_test = get_class_i_indices(y_test, classDict['bird'])
    cat_indices_test = get_class_i_indices(y_test, classDict['cat'])
    deer_indices_test = get_class_i_indices(y_test, classDict['deer'])
    dog_indices_test = get_class_i_indices(y_test, classDict['dog'])
    frog_indices_test = get_class_i_indices(y_test, classDict['frog'])
    horse_indices_test = get_class_i_indices(y_test, classDict['horse'])
    ship_indices_test = get_class_i_indices(y_test, classDict['ship'])
    truck_indices_test = get_class_i_indices(y_test, classDict['truck'])

    subset_indices_1 = plane_indices[0:500] + car_indices[0:500] + bird_indices[0:500] + cat_indices[0:500] + deer_indices[
                                                                                                          0:500] + dog_indices[
                                                                                                                   0:500] + frog_indices[
                                                                                                                            0:500] + horse_indices[
                                                                                                                                     0:500] + ship_indices[
                                                                                                                                              0:500] + truck_indices[
                                                                                                                                                       0:500]

    subset_indices_test_1 = plane_indices_test[0:100] + car_indices_test[0:100] + bird_indices_test[
                                                                              0:100] + cat_indices_test[
                                                                                       0:100] + deer_indices_test[
                                                                                                0:100] + dog_indices_test[
                                                                                                         0:100] + frog_indices_test[
                                                                                                                  0:100] + horse_indices_test[
                                                                                                                           0:100] + ship_indices_test[
                                                                                                                                    0:100] + truck_indices_test[
                                                                                                                                             0:100]
    return subset_indices_1, subset_indices_test_1
    trainset_1 = torch.utils.data.Subset(trainset, subset_indices_1)
    testset_1 = torch.utils.data.Subset(testset, subset_indices_test_1)

