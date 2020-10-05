import argparse
import os
import numpy as np
from multiprocessing import Pool


def float2uint(file_path):
	filename = file_path[:file_path.find('.npy')]+'.npz'
	data_float32 = np.load(file_path)
	data_temp = 255 * data_float32
	data_uint8 = data_temp.astype(np.uint8)
	print(filename)
	np.savez_compressed(filename, data=data_uint8)
	os.remove(file_path)

parser = argparse.ArgumentParser()
parser.add_argument('-output', '--output', default='/home/pkao/brats2017-master/output', type=str)
#parser.add_argument('-output', '--output', default='/media/hdd1/pkao/brats2018/output', type=str)
parser.add_argument('-cfg', '--cfg', default='unet_ce_hard', type=str)
#parser.add_argument('-mode', '--mode', default='validation', type=str)

args = parser.parse_args()

#root_dir = os.path.join(args.output, args.mode, args.cfg)
root_dir = os.path.join(args.output, args.cfg)
#print(root_dir)

float32_paths = [os.path.join(root, name) for root, dirs, files in os.walk(root_dir) for name in files if name.endswith('.npy')]

assert(len(float32_paths) == 191)
pool = Pool(16)

pool.map(float2uint, float32_paths)

