#echo $OMP_NUM_THREADS
import pickle
import time
import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Spatial, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad
from .transforms import NumpyType

from .data_utils import pkload, gen_feats

import numpy as np


class SingleData28(Dataset):
    def __init__(self, list_file, root='', for_train=False,
            transforms='', return_target=True, crop=True):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.return_target = return_target

        self.transforms   = eval(transforms or 'Identity()')
        self.feats = gen_feats()

    def __getitem__(self, index):
        path = self.paths[index]

        x, y = pkload(path + 'data_f32.pkl')
        x = np.concatenate([x, self.feats], -1)

        mask = np.load(path + 'HarvardOxford-sub.npy')
        # transforms work with nhwtc
        x, y, mask = x[None, ...], y[None, ...], mask[None, ...]

        done = False
        if self.return_target:
            while not done:
                a, b, c = self.transforms([x, y, mask])
                if b.sum() > 0:
                    done = True
                    x, y, mask = a, b, c
        else:
            x, mask = self.transforms([x, mask])
            y = np.array([1])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)

        x, y, mask = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)

        return x, y, mask


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

    #@staticmethod
    #def add_mask(x, mask, dim=1):
    #    mask = mask.unsqueeze(dim)
    #    shape = list(x.shape); shape[dim] += 21
    #    new_x = x.new(*shape).zero_()
    #    new_x = new_x.scatter_(dim, mask, 1.0)
    #    s = [slice(None)]*len(shape)
    #    s[dim] = slice(21, None)
    #    new_x[s] = x
    #    return new_x


class SingleData25(Dataset):
    def __init__(self, list_file, root='', for_train=False,
            transforms='', return_target=True, crop=True):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.return_target = return_target

        self.transforms   = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]

        x, y = pkload(path + 'data_f32.pkl')
        mask = np.load(path + 'HarvardOxford-sub.npy')

        # transforms work with nhwtc
        x, y, mask = x[None, ...], y[None, ...], mask[None, ...]

        done = False
        if self.return_target:
            while not done:
                a, b, c = self.transforms([x, y, mask])
                if b.sum() > 0:
                    done = True
                    x, y, mask = a, b, c
        else:
            x, mask = self.transforms([x, mask])
            y = np.array([1])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)

        x, y, mask = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)

        return x, y, mask

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

    #@staticmethod
    #def add_mask(x, mask, dim=1):
    #    mask = mask.unsqueeze(dim)
    #    shape = list(x.shape); shape[dim] += 21
    #    new_x = x.new(*shape).zero_()
    #    new_x = new_x.scatter_(dim, mask, 1.0)
    #    s = [slice(None)]*len(shape)
    #    s[dim] = slice(21, None)
    #    new_x[s] = x
    #    return new_x


class SingleData(Dataset):
    def __init__(self, list_file, root='', for_train=False,
            transforms='', return_target=True, crop=True):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.return_target = return_target

        self.transforms   = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]

        x, y = pkload(path + 'data_f32.pkl')

        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]

        done = False
        if self.return_target:
            while not done:
                a, b = self.transforms([x, y])
                if b.sum() > 0:
                    done = True
                    x, y = a, b

        else:
            x = self.transforms(x)

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


#S = '''Compose([
#    RandCrop(128),
#    Rot90(axes=(0,1)),
#    ])'''
#
#S = '''Compose([
#    Pad((0, 0, 5, 0)),
#    Rot90(axes=(0,1)),
#    ])'''
#
#root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
#file_list = root + 'all.txt'
##dset = SingleData(file_list, root=root, for_train=True)
#dset = SingleData(file_list, root=root, for_train=True, geo_transforms=S)
#print(dset.paths[0])
#print(dset.names[0])
#x, y = dset[0]
#print(x.shape, y.shape)
#exit(0)
#import time
#start = time.time()
##for i in range(len(dset)):
#for i in range(10):
#    dset[i]
#    #x1, x2, y, c = dset[0]
#    print(time.time() - start)
#    start = time.time()
#
