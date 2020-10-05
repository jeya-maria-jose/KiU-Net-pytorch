import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim
cudnn.benchmark = True

import multicrop
import numpy as np
import nibabel as nib
from medpy import metric

import models
from models import criterions
from data import datasets
from data.data_utils import add_mask
from utils import Parser

path = os.path.dirname(__file__)

def calculate_metrics(pred, target):
    sens = metric.sensitivity(pred, target)
    spec = metric.specificity(pred, target)
    dice = metric.dc(pred, target)

eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

#https://github.com/ellisdg/3DUnetCNN
#https://github.com/ellisdg/3DUnetCNN/blob/master/brats/evaluate.py
#https://github.com/MIC-DKFZ/BraTS2017
#https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py
def dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0
    ret += f1_score(o, t),
    # core
    o = (output==1) | (output==4)
    t = (target==1) | (target==4)
    ret += f1_score(o , t),
    # active
    o = (output==4); t = (target==4)
    ret += f1_score(o , t),

    return ret

keys = 'whole', 'core', 'enhancing', 'loss'
def main():
    ckpts = args.getdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # setup networks
    Network = getattr(models.unet, args.net)
    # model = models.unet.kiunet()
    model = Network(**args.net_params)
    model = model.cuda()

    model_file = os.path.join(ckpts, args.ckpt)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    Dataset = getattr(datasets, args.dataset)

    valid_list = os.path.join(args.data_dir, args.valid_list)
    valid_set = Dataset(valid_list, root=args.data_dir,
            for_train=False, return_target=args.scoring,
            transforms=args.test_transforms)
    valid_loader = DataLoader(
        valid_set,
        batch_size=1, shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=1, pin_memory=True)

    start = time.time()
    with torch.no_grad():
        scores = validate(valid_loader, model,
                args.out_dir, valid_set.names, scoring=args.scoring)

    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def validate(valid_loader, model,
        out_dir='', names=None, scoring=True, verbose=True):

    H, W, T = 240, 240, 155
    dtype = torch.float32

    dset = valid_loader.dataset

    model.eval()
    criterion = F.cross_entropy

    vals = AverageMeter()
    for i, data in enumerate(valid_loader):

        # target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None
        data = [t.cuda(non_blocking=True) for t in data]

        x, target = data[:2]

        if len(data) > 2:
            x = add_mask(x, data.pop(), 1)

        # compute output
        logit = model(x) # nx5x9x9x9, target nx9x9x9
        output = F.softmax(logit, dim=1) # nx5x9x9x9

        ## measure accuracy and record loss
        #loss = None
        #if scoring and criterion is not None:
        #    loss = criterion(logit, target).item()

        output = output[0, :, :H, :W, :T].cpu().numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            oname = os.path.join(out_dir, name + '.nii.gz')
            H, W, T = 240, 240, 155
            seg_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
            output = output.argmax(0)
            # print(seg_img.shape,output.shape)
            # ET_voxels = (output == 4).sum()
            # if ET_voxels < 500:
            #     output[np.where(output == 4)] = 1
            seg_img[np.where(output==1)] = 1
            seg_img[np.where(output==2)] = 2
            seg_img[np.where(output==4)] = 4
            
            # if verbose:
            print('1:',np.sum(seg_img==1),' | 2:',np.sum(seg_img==2),' | 4:',np.sum(seg_img==4))
            print('WT:',np.sum((seg_img==1)|(seg_img==2)|(seg_img==4)),' | TC:',np.sum((seg_img==1)|(seg_img==4)),' | ET:',np.sum(seg_img==4))
            nib.save(nib.Nifti1Image(seg_img, None),oname)

        if scoring:
            output = output.argmax(0)
            scores = dice(output, target_cpu)

            #if loss is not None:
            #    scores += loss,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if verbose:
            logging.info(msg)

    if scoring:
        msg = 'Average scores: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
        logging.info(msg)

    model.train()
    return vals.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', default='unet', type=str)
    parser.add_argument('-gpu', '--gpu', default='0', type=str)
    args = parser.parse_args()

    args = Parser(args.cfg, log='test').add_args(args)

    #args.valid_list = 'valid_0.txt'
    #args.valid_list = 'all.txt'
    #args.saving = False
    #args.scoring = True

    args.data_dir = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/brats/2019/val'
    args.valid_list = 'valid.txt'
    args.saving = True
    args.scoring = False # for real test data, set this to False

    args.ckpt = 'model_epoch_100.tar'
    #args.ckpt = 'model_iter_227.tar'

    if args.saving:
        folder = os.path.splitext(args.valid_list)[0]
        out_dir = os.path.join('output', args.name, folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
    else:
        args.out_dir = ''


    main()