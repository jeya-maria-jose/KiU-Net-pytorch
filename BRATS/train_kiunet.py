import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
cudnn.benchmark = True

import numpy as np

import models
from models import criterions
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import add_mask, init_fn
from utils import Parser

from predict_unet import validate, AverageMeter

parser = argparse.ArgumentParser()
#parser.add_argument('-cfg', '--cfg', default='unet_all', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_dice2_c25', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_dice2_redo', type=str)
parser.add_argument('-cfg', '--cfg', default='unet_ce_hard', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_bce_hard_per_im', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_bce_mean', type=str)
parser.add_argument('-gpu', '--gpu', default='1', type=str)
parser.add_argument('-out', '--out', default='', type=str)

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
args.gpu = str(args.gpu)

ckpts = args.makedir()
resume = os.path.join(ckpts, 'model_last.tar')
#resume = ''
if not args.resume and os.path.exists(resume):
    args.resume = resume

def main():
    # setup environments and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup networks
    # Network = getattr(models, args.net)
    # model = Network(**args.net_params)
    model = models.unet.kiunet()
    model = model.cuda()

    optimizer = getattr(torch.optim, args.opt)(
            model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    msg = ''
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)
    logging.info(msg)

    # Data loading code
    Dataset = getattr(datasets, args.dataset)

    train_list = os.path.join(args.data_dir, args.train_list)
    train_set = Dataset(train_list, root=args.data_dir, for_train=True,
            transforms=args.train_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, worker_init_fn=init_fn)

    if args.valid_list:
        valid_list = os.path.join(args.data_dir, args.valid_list)
        valid_set = Dataset(valid_list, root=args.data_dir,
                for_train=False, transforms=args.test_transforms)

        valid_loader = DataLoader(
            valid_set, batch_size=1, shuffle=False,
            collate_fn=valid_set.collate,
            num_workers=4, pin_memory=True)

        train_valid_set = Dataset(train_list, root=args.data_dir,
                for_train=False, transforms=args.test_transforms)

        train_valid_loader = DataLoader(
            train_valid_set, batch_size=1, shuffle=False,
            collate_fn=train_valid_set.collate,
            num_workers=4, pin_memory=True)


    start = time.time()

    enum_batches = len(train_set)/float(args.batch_size)
    args.schedule   = {int(k*enum_batches): v for k, v in args.schedule.items()}
    args.save_freq  = int(enum_batches * args.save_freq)
    args.valid_freq = int(enum_batches * args.valid_freq)

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    for i, data in enumerate(train_loader, args.start_iter):
        ## validation
        #if args.valid_list and  (i % args.valid_freq) == 0:
        #    logging.info('-'*50)
        #    msg  =  'Iter {}, Epoch {:.4f}, {}'.format(i, i/enum_batches, 'validation')
        #    logging.info(msg)
        #    with torch.no_grad():
        #        validate(valid_loader, model, names=valid_set.names)

        # actual training
        adjust_learning_rate(optimizer, i)

        data = [t.cuda(non_blocking=True) for t in data]

        x, target = data[:2]

        if len(data) > 2: # has mask
            x = add_mask(x, data.pop(), 1)

        # logging.info(x.shape)
        # logging.info(torch.max(x))
        # logging.info(torch.min(x))
        # logging.info(target.shape)
        # logging.info(torch.max(target))
        # logging.info(torch.min(target))

        # compute output
        output = model(x) # nx5x128x128x128, target nx128x128x128
        loss = criterion(output, target, args.alpha)

        # measure accuracy and record loss
        losses.update(loss.item(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.save_freq == 0:
            epoch = int((i+1) // enum_batches)
            file_name = os.path.join(ckpts, 'model_epoch_{}.tar'.format(epoch))
            torch.save({
                'iter': i+1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.4f}'.format(
                i+1, (i+1)/enum_batches, losses.avg)
        logging.info(msg)

        losses.reset()

    i = num_iters + args.start_iter
    file_name = os.path.join(ckpts, 'model_last.tar')
    torch.save({
        'iter': i,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        },
        file_name)

    if args.valid_list:
        logging.info('-'*50)
        msg  =  'Iter {}, Epoch {:.4f}, {}'.format(i, i/enum_batches, 'validate validation data')
        logging.info(msg)
        with torch.no_grad():
            validate(valid_loader, model, names=valid_set.names, out_dir=args.out)

        #logging.info('-'*50)
        #msg  =  'Iter {}, Epoch {:.4f}, {}'.format(i, i/enum_batches, 'validate training data')
        #logging.info(msg)
        #with torch.no_grad():
        #    validate(train_valid_loader, model, names=train_valid_set.names, verbose=False)

    msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch):
    # reduce learning rate by a factor of 10
    if epoch+1 in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()
