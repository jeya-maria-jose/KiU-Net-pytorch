import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
import nibabel as nib

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data().astype('float32')
    proxy.uncache()
    return data

def crop(x, ksize, stride=3):
    shape = (np.array(x.shape[:3]) - ksize)/stride + 1
    shape = tuple(shape) + (ksize, )*3 + (x.shape[3], )

    strides = np.array(x.strides[:3])*3
    strides = tuple(strides) + x.strides

    x = as_strided(x, shape=shape, strides=strides)
    return x



modalities = ('flair', 't1ce', 't1', 't2')
root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
file_list = root + 'file_list.txt'
subjects = open(file_list).read().splitlines()
subj = subjects[0]
name = subj.split('/')[-1]
path = os.path.join(root, subj, name + '_')

x0 = np.stack([
    nib_load(path + modal + '.nii.gz') \
    for modal in modalities], 3)
y0 = nib_load(path + 'seg.nii.gz')[..., None]

x0 = np.pad(x0, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')
y0 = np.pad(y0, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')

x1 = crop(x0, 9)
x2 = crop(np.pad(x0, ((8, 8), (8, 8), (8, 8), (0, 0)), mode='constant'), 25)
x3 = crop(np.pad(x0, ((24, 24), (24, 24), (24, 24), (0, 0)), mode='constant'), 57)

y1 = crop(y0, 9)

m = x1.reshape(x1.shape[:3] + (-1, )).sum(3) > 0
x1 = x1[m]
x2 = x2[m]
x3 = x3[m]
y1 = y1[m]
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(y1.shape)

