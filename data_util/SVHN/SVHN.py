import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import svhn_truncated


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)




class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_svhn():
    CIFAR_MEAN = [0.4377, 0.4438, 0.4728]
    CIFAR_STD = [0.1201, 0.1231, 0.1052]

    train_transform = transforms.Compose([

        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_svhn_data(datadir):
    train_transform, test_transform = _data_transforms_svhn()

    svhn_train_ds = svhn_truncated(datadir, split="train", transform=train_transform, download=True)
    svhn_test_ds = svhn_truncated(datadir, split="test", transform=test_transform, download=True)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data_dataset(X_train, y_train, n_nets, alpha,partition_rate=0):
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}
    indices = np.random.permutation(N)
    split_index = int(partition_rate * N)
    y_train_public_index = list(indices)[:split_index]

    while min_size < 10:
        # print(min_size)
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset


        for k in range(K):
            # print("K",k)
            idx_k = np.where(y_train == k)[0]
            idx_k = np.setdiff1d(idx_k, y_train_public_index)

            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

            np.random.shuffle(idx_k)

            # print("proportions1",proportions)
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            # print("proportions2",proportions)
            proportions = proportions / proportions.sum()
            # print("proportions3",proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # print("proportions4",proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    net_dataidx_map_train_public = y_train_public_index
    return net_dataidx_map,net_dataidx_map_train_public


def partition_data(dataset, datadir, partition, n_nets, alpha, partition_rate=0):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num = n_test
        idxs = np.random.permutation(total_num)
        idxs_test = np.random.permutation(test_total_num)
        if partition_rate:
            batch_idxs = np.array_split(idxs[:int(total_num*(1-partition_rate))], n_nets)
        else:
            batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_train_public=idxs[int(total_num*(1 - partition_rate)):]
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":
        net_dataidx_map_train, net_dataidx_map_train_public = partition_data_dataset(X_train, y_train, n_nets, alpha,
                                                                             partition_rate)
        net_dataidx_map_test = partition_data_dataset(X_test, y_test, n_nets, alpha)[0]

    else:
        raise Exception("partition arg error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, net_dataidx_map_train_public


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    return get_dataloader_svhn(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_svhn(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_svhn(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = svhn_truncated

    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, split="train", transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, split="test", transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_svhn(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = svhn_truncated

    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, split="train", transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, split="test", transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_svhn(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,partition_rate):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, net_dataidx_map_train_public = partition_data(dataset,
                                                                                                   data_dir,
                                                                                                   partition_method,
                                                                                                   client_number,
                                                                                                   partition_alpha,
                                                                                                   partition_rate)
    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    trian_data_public=get_dataloader(dataset,data_dir,batch_size,batch_size,dataidxs_train=net_dataidx_map_train_public)[0]


    # get local dataset
    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test = net_dataidx_map_test[client_idx]

        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)

        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test


        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs_train, dataidxs_test)


        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, train_data_global, test_data_global,trian_data_public, \
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test
