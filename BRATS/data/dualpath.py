#echo $OMP_NUM_THREADS
import pickle
import time
import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Spatial, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import ToNumpy, NumpyType, ToTensor, TensorType

from .data_utils import sample, pkload, get_all_coords, gen_feats, _zero, _shape

import random

import numpy as np
import multicrop


class DualData28(Dataset):
    def __init__(self, list_file, root='', num_patches=20, for_train=False,
            transforms='', return_target=True, crop=True,
            sample_size=25, sub_sample_size=19, target_size=19):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.root = root
        self.names = names
        self.paths = paths
        self.crop = crop
        self.num_patches = num_patches
        self.for_train = for_train
        self.return_target = return_target

        self.sample_size = sample_size
        self.sub_sample_size = sub_sample_size
        self.target_size = target_size

        self.suffix = '{}x{}x{}_'.format(sample_size, sample_size, sample_size)

        self.all_coords = get_all_coords(target_size)
        self.shape = np.ceil(np.array(_shape, dtype='float32')/target_size).astype('int')

        self.transforms   = eval(transforms or 'Identity()')

        self.feats = gen_feats()

    def __getitem__(self, index):
        path = self.paths[index]

        # faster than niffty
        #images, label = pkload(path + 'data_f32_divm.pkl')

        #images, label = pkload(path + 'data_f32.pkl')
        #images, label = torch.tensor(images), torch.tensor(label)

        images, label = pkload(path + 'data_f32.pkl')
        images, label = torch.from_numpy(images), torch.from_numpy(label)

        images = np.concatenate([images, self.feats], -1)

        mask = np.load(path + 'HarvardOxford-sub.npy')
        mask = torch.from_numpy(mask)

        if not self.crop:
            # transformation needs nhwtc
            images, label, mask = images.unsqueeze(0), label.unsqueeze(0), mask.unsqueeze(0)
            images, label, mask = self.transforms([images, label, mask])
            images, label, mask = images.squeeze(0), label.squeeze(0), mask.squeeze(0)
            images = images.permute(3, 0, 1, 2).contiguous()

            return (images, self.all_coords, mask), label

        if self.for_train:
            fg, bg = pkload(path + self.suffix + 'coords.pkl')
            coords = torch.cat([sample(x, self.num_patches//2) for x in (fg, bg)])
        else:
            coords = self.all_coords

        samples = multicrop.crop3d_cpu(images, coords,
                self.sample_size, self.sample_size, self.sample_size, 1, False)

        sub_samples = multicrop.crop3d_cpu(images, coords,
                self.sub_sample_size, self.sub_sample_size, self.sub_sample_size, 3, False)

        mask_id = multicrop.crop3d_cpu(mask, coords,
                self.sample_size, self.sample_size, self.sample_size, 1, False)

        sub_mask_id = multicrop.crop3d_cpu(mask, coords,
                self.sub_sample_size, self.sub_sample_size, self.sub_sample_size, 3, False)

        if self.return_target:
            target = multicrop.crop3d_cpu(
                        label, coords,
                        self.target_size, self.target_size, self.target_size, 1, False)
            samples, sub_samples, mask_id, sub_mask_id, target = self.transforms([samples, sub_samples, mask_id, sub_mask_id, target])
        else:
            samples, sub_samples, mask_id, sub_mask_id = self.transforms([samples, sub_samples, mask_id, sub_mask_id])
            target = coords

        if self.for_train: label = _zero

        samples = samples.permute(0, 4, 1, 2, 3).contiguous()
        sub_samples = sub_samples.permute(0, 4, 1, 2, 3).contiguous()

        #samples = self.add_mask(samples, mask_id, 1)
        #sub_samples = self.add_mask(sub_samples, sub_mask_id, 1)

        return (samples, sub_samples, target, mask_id, sub_mask_id), label

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

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        data, label = list(zip(*batch))
        data  = [torch.cat(v) for v in zip(*data)]
        label = torch.cat(label)

        if self.for_train:
            perm = torch.randperm(data[0].shape[0])
            data = [t[perm] for t in data]

        return data, label


class DualData25(Dataset):
    def __init__(self, list_file, root='', num_patches=20, for_train=False,
            transforms='', return_target=True, crop=True,
            sample_size=25, sub_sample_size=19, target_size=19):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.root = root
        self.names = names
        self.paths = paths
        self.crop = crop
        self.num_patches = num_patches
        self.for_train = for_train
        self.return_target = return_target

        self.sample_size = sample_size
        self.sub_sample_size = sub_sample_size
        self.target_size = target_size

        self.suffix = '{}x{}x{}_'.format(sample_size, sample_size, sample_size)

        self.all_coords = get_all_coords(target_size)
        self.shape = np.ceil(np.array(_shape, dtype='float32')/target_size).astype('int')

        self.transforms   = eval(transforms or 'Identity()')


    def __getitem__(self, index):
        path = self.paths[index]

        # faster than niffty
        #images, label = pkload(path + 'data_f32_divm.pkl')

        #images, label = pkload(path + 'data_f32.pkl')
        #images, label = torch.tensor(images), torch.tensor(label)

        images, label = pkload(path + 'data_f32.pkl')
        images, label = torch.from_numpy(images), torch.from_numpy(label)


        mask = np.load(path + 'HarvardOxford-sub.npy')
        mask = torch.from_numpy(mask)

        if not self.crop:
            # transformation needs nhwtc
            images, label, mask = images.unsqueeze(0), label.unsqueeze(0), mask.unsqueeze(0)
            images, label, mask = self.transforms([images, label, mask])
            images, label, mask = images.squeeze(0), label.squeeze(0), mask.squeeze(0)
            images = images.permute(3, 0, 1, 2).contiguous()

            return (images, self.all_coords, mask), label

        if self.for_train:
            fg, bg = pkload(path + self.suffix + 'coords.pkl')
            coords = torch.cat([sample(x, self.num_patches//2) for x in (fg, bg)])
        else:
            coords = self.all_coords

        samples = multicrop.crop3d_cpu(images, coords,
                self.sample_size, self.sample_size, self.sample_size, 1, False)

        sub_samples = multicrop.crop3d_cpu(images, coords,
                self.sub_sample_size, self.sub_sample_size, self.sub_sample_size, 3, False)

        mask_id = multicrop.crop3d_cpu(mask, coords,
                self.sample_size, self.sample_size, self.sample_size, 1, False)

        sub_mask_id = multicrop.crop3d_cpu(mask, coords,
                self.sub_sample_size, self.sub_sample_size, self.sub_sample_size, 3, False)

        if self.return_target:
            target = multicrop.crop3d_cpu(
                        label, coords,
                        self.target_size, self.target_size, self.target_size, 1, False)
            samples, sub_samples, mask_id, sub_mask_id, target = self.transforms([samples, sub_samples, mask_id, sub_mask_id, target])
        else:
            samples, sub_samples, mask_id, sub_mask_id = self.transforms([samples, sub_samples, mask_id, sub_mask_id])
            target = coords

        if self.for_train: label = _zero

        samples = samples.permute(0, 4, 1, 2, 3).contiguous()
        sub_samples = sub_samples.permute(0, 4, 1, 2, 3).contiguous()

        #samples = self.add_mask(samples, mask_id, 1)
        #sub_samples = self.add_mask(sub_samples, sub_mask_id, 1)

        return (samples, sub_samples, target, mask_id, sub_mask_id), label

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

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        data, label = list(zip(*batch))
        data  = [torch.cat(v) for v in zip(*data)]
        label = torch.cat(label)

        if self.for_train:
            perm = torch.randperm(data[0].shape[0])
            data = [t[perm] for t in data]

        return data, label


class DualData(Dataset):
    def __init__(self, list_file, root='', num_patches=20, for_train=False,
            transforms='', return_target=True, crop=True,
            sample_size=25, sub_sample_size=19, target_size=19):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.root = root
        self.names = names
        self.paths = paths
        self.crop = crop
        self.num_patches = num_patches
        self.for_train = for_train
        self.return_target = return_target

        self.sample_size = sample_size
        self.sub_sample_size = sub_sample_size
        self.target_size = target_size

        self.suffix = '{}x{}x{}_'.format(sample_size, sample_size, sample_size)

        self.all_coords = get_all_coords(target_size)
        self.shape = np.ceil(np.array(_shape, dtype='float32')/target_size).astype('int')

        self.transforms   = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]

        # faster than niffty
        #images, label = pkload(path + 'data_f32_divm.pkl')

        #images, label = pkload(path + 'data_f32.pkl')
        #images, label = torch.tensor(images), torch.tensor(label)

        images, label = pkload(path + 'data_f32.pkl')
        images, label = torch.from_numpy(images), torch.from_numpy(label)


        if not self.crop:
            # transformation needs nhwtc
            images, label = images.unsqueeze(0), label.unsqueeze(0)
            images, label = self.transforms([images, label])
            images, label = images.squeeze(0), label.squeeze(0)

            images = images.permute(3, 0, 1, 2).contiguous()
            return (images, self.all_coords), label

        if self.for_train:
            fg, bg = pkload(path + self.suffix + 'coords.pkl')
            coords = torch.cat([sample(x, self.num_patches//2) for x in (fg, bg)])
        else:
            coords = self.all_coords

        samples = multicrop.crop3d_cpu(images, coords,
                self.sample_size, self.sample_size, self.sample_size, 1, False)

        sub_samples = multicrop.crop3d_cpu(images, coords,
                self.sub_sample_size, self.sub_sample_size, self.sub_sample_size, 3, False)

        if self.return_target:
            target = multicrop.crop3d_cpu(
                        label, coords,
                        self.target_size, self.target_size, self.target_size, 1, False)
            samples, sub_samples, target = self.transforms([samples, sub_samples, target])
        else:
            samples, sub_samples = self.transforms([samples, sub_samples])
            target = coords

        if self.for_train: label = _zero

        samples = samples.permute(0, 4, 1, 2, 3).contiguous()
        sub_samples = sub_samples.permute(0, 4, 1, 2, 3).contiguous()
        return (samples, sub_samples, target), label

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        data, label = list(zip(*batch))
        data  = [torch.cat(v) for v in zip(*data)]
        label = torch.cat(label)

        if self.for_train:
            perm = torch.randperm(data[0].shape[0])
            data = [t[perm] for t in data]

        return data, label

#root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
#file_list = root + 'all.txt'
##dset = DualData(file_list, root=root, for_train=True)
#dset = DualData(file_list, root=root, for_train=False)
#print(dset.paths[0])
#print(dset.names[0])
#import time
#start = time.time()
##for i in range(len(dset)):
#for i in range(10):
#    dset[i]
#    #x1, x2, y, c = dset[0]
#    print(time.time() - start)
#    start = time.time()

