import math
import random
import collections
import numpy as np
import torch
from scipy import ndimage

from .rand import Constant, Uniform, Gaussian

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Itendity()'


Identity = Base


# gemetric transformations, need a buffers
# first axis is N
class Rot90(Base):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

        for a in self.axes:
            assert a > 0

    def sample(self, *shape):
        shape = list(shape)
        i, j = self.axes

        # shape: no first dim
        i, j = i-1, j-1
        shape[i], shape[j] = shape[j], shape[i]

        return shape

    def tf(self, img, k=0):
        return np.rot90(img, axes=self.axes)

    def __str__(self):
        return 'Rot90(axes=({}, {})'.format(*self.axes)


class Flip(Base):
    def __init__(self, axis=0):
        self.axis = axis

    def tf(self, img, k=0):
        return np.flip(img, self.axis)

    def __str__(self):
        return 'Flip(axis={})'.format(self.axis)


class RandSelect(Base):
    def __init__(self, prob=0.5, tf=None):
        self.prob = prob
        self.ops  = tf if isinstance(tf, collections.Sequence) else (tf, )
        self.buff = False

    def sample(self, *shape):
        self.buff = random.random() < self.prob

        if self.buff:
            for op in self.ops:
                shape = op.sample(*shape)

        return shape

    def tf(self, img, k=0):
        if self.buff:
            for op in self.ops:
                img = op.tf(img, k)
        return img

    def __str__(self):
        if len(self.ops) == 1:
            ops = str(self.ops[0])
        else:
            ops = '[{}]'.format(', '.join([str(op) for op in self.ops]))
        return 'RandSelect({}, {})'.format(self.prob, ops)


class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s -size)//2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size] * len(shape)


    def tf(self, img, k=0):
        return img[self.buffer]

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)


class RandCrop(CenterCrop):
    def sample(self, *shape):
        size = self.size
        start = [random.randint(0, s-size) for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size]*len(shape)

    def __str__(self):
        return 'RandCrop({})'.format(self.size)


class Pad(Base):
    def __init__(self, pad):
        self.pad = pad
        self.px = tuple(zip([0]*len(pad), pad))

    def sample(self, *shape):

        shape = list(shape)

        # shape: no first dim
        for i in range(len(shape)):
            shape[i] += self.pad[i+1]

        return shape

    def tf(self, img, k=0):
        #nhwtc, nhwt
        dim = len(img.shape)
        return np.pad(img, self.px[:dim], mode='constant')

    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)

# for data only
## No buffers, color transformation
class Noise(Base):
    def __init__(self, dim, sigma=0.1, channel=True, num=-1):
        self.dim = dim
        self.sigma = sigma
        self.channel = channel
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        if self.channel:
            #nhwtc, hwtc, hwt
            shape = [1] if len(img.shape) < self.dim+2 else [img.shape[-1]]
        else:
            shape = img.shape
        return img * np.exp(self.sigma * torch.randn(shape, dtype=torch.float32).numpy())

    def __str__(self):
        return 'Noise()'


# dim could come from shape
class GaussianBlur(Base):
    def __init__(self, dim, sigma=Constant(1.5), app=-1):
        # 1.5 pixel
        self.dim = dim
        self.sigma = sigma
        self.eps   = 0.001
        self.app = app

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        # image is nhwtc
        for n in range(img.shape[0]):
            sig = self.sigma.sample()
            # sample each channel saperately to avoid correlations
            if sig > self.eps:
                if len(img.shape) == self.dim+2:
                    C = img.shape[-1]
                    for c in range(C):
                        img[n,..., c] = ndimage.gaussian_filter(img[n, ..., c], sig)
                elif len(img.shape) == self.dim+1:
                    img[n] = ndimage.gaussian_filter(img[n], sig)
                else:
                    raise ValueError('image shape is not supported')

        return img

    def __str__(self):
        return 'GaussianBlur()'


class ToNumpy(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.numpy()

    def __str__(self):
        return 'ToNumpy()'


class ToTensor(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        return torch.from_numpy(img)

    def __str__(self):
        return 'ToTensor'


class TensorType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('torch.float32', 'torch.int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.type(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'TensorType(({}))'.format(s)


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0, num=-1):
        self.mean = mean
        self.std = std
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        img -= self.mean
        img /= self.std
        return img

    def __str__(self):
        return 'Normalize()'


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)


# adapt from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/spatial_transformations.py#L25
from .tf_utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords

class Spatial(Base):
    def __init__(self, patch_size, center_to_border=None,
            alpha=None, sigma=None,
            angle_x=None, angle_y=None, angle_z=None,
            scale=None, random_crop=True):

        self.patch_size = patch_size
        self.buff = None
        self.alpha = alpha
        self.sigma = sigma
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.scale = scale
        self.dim = len(patch_size)
        self.random_crop = random_crop

        if center_to_border is None:
            center_to_border = list(np.array(patch_size)//2)
        elif not isinstance(center_to_border, collections.Sequence):
            center_to_border = self.dim * (center_to_border, )
        else:
            raise ValueError('center to border')

        self.center_to_border = center_to_border


        self.order_data = 3
        self.border_mode_data = 'nearest'
        self.border_cval_data = 0

    def sample(self, *shape):
        #nhwtc
        coords = create_zero_centered_coordinate_mesh(self.patch_size)
        # dimxhxwxt

        if self.alpha is not None and self.sigma is not None:
            a = self.alpha.sample()
            s = self.sigma.sample()
            coords = elastic_deform_coordinates(coords, a, s)

        if self.angle_x is not None:
            ax = self.angle_x.sample()
            if self.dim == 3:
                ay = self.angle_y.sample()
                az = self.angle_z.sample()
                coords = rotate_coords_3d(coords, ax, ay, az)
            else:
                coords = rotate_coords_2d(coords, ax)
        if self.scale is not None:
            sc = self.scale.sample()
            coords = scale_coords(coords, sc)

        for d in range(self.dim):
            if self.random_crop:
                ctr = random.uniform(self.center_to_border[d],
                                        shape[d] - self.center_to_border[d])
            else:
                ctr = int(np.round(shape[d] / 2.))
            coords[d] += ctr

        self.buff = coords

        return self.patch_size

    def tf(self, img):
        shape = list(img.shape)
        shape[1:self.dim+1] = self.patch_size
        out = np.zeros(shape, dtype=img.dtype)
        for n in range(img.shape[0]):
            if len(img.shape) == self.dim+2:
                for c in range(img.shape[-1]):
                    out[n, ..., c] = interpolate_img(img[n, ..., c], self.buff,
                            self.order_data, self.border_mode_data, cval=self.border_cval_data)
            elif len(img.shape) == self.dim+1:
                out[n] = interpolate_img(img[n], self.buff,
                        self.order_data, self.border_mode_data, cval=self.border_cval_data)
            else:
                raise ValueError('image shape is not supported')
        return out

