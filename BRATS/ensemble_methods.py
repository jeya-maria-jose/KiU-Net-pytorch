import os
import numpy as np
import nibabel as nib

settings= {
	'test': {
	'models':['deepmedic_ce', 'unet_dice', 'deepmedic_ce_50_50_c25', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50', 'deepmedic_ce_90_120_150_b50_mb50']
	},

	'test_1':{
	'models':['deepmedic_ce_50_50_c25_noaug']
	},

	'ensemble_8': {
	'models': ['deepmedic_ce', 'unet_dice', 'deepmedic_ce_50_50_c25', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50', 'deepmedic_ce_90_120_150_b50_mb50', 'deepmedic_ce_50_50_c25_noaug']
	},

	'ensemble_26':{
	'models': ['deepmedic_ce', 'unet_dice', 'deepmedic_ce_50_50_c25', 'deepmedic_ce_50_50_aug', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50', 'deepmedic_ce_90_120_150_b50_mb50', 'deepmedic_ce_50_50_c25_noaug', 'deepmedic_ce_c25_60_80_100_b50_mb50', 'deepmedic_ce_c25_90_120_150_b50_mb50', 'deepmedic_ce_c25_45_60_75_b50_mb50', 'deepmedic_ce_c25_75_100_125_b50_mb50', 'deepmedic_ce_aug', 'deepmedic_ce_50_50', 'deepmedic_ce_90_120_150_b50_mb50_aug', 'deepmedic_ce_60_80_100_b50_mb50_aug', 'deepmedic_ce_75_100_125_b50_mb50_aug', 'deepmedic_ce_22x18x6_aug', 'deepmedic_ce_75_100_125_b50_mb50', 'deepmedic_ce_28x20x12_aug', 'deepmedic_ce_45_60_75_b50_mb50', 'munet_dice', 'unet_dice_c25', 'unet_ce_hard_c25', 'unet_ce_hard_per_im_c25']
	}, 
	'ensemble_26_testing':{
	'models': ['deepmedic_ce_all', 'unet_dice_all', 'deepmedic_ce_50_50_c25_all', 
	'deepmedic_ce_50_50_all_aug', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50_all',
	'deepmedic_ce_90_120_150_b50_mb50_all', 'deepmedic_ce_50_50_c25_all_noaug', 'deepmedic_ce_c25_60_80_100_b50_mb50_all', 
	'deepmedic_ce_c25_90_120_150_b50_mb50_all', 'deepmedic_ce_c25_45_60_75_b50_mb50_all', 
	'deepmedic_ce_c25_75_100_125_b50_mb50_all', 'deepmedic_ce_all_aug', 'deepmedic_ce_50_50_all', 
	'deepmedic_ce_90_120_150_b50_mb50_all_aug', 'deepmedic_ce_60_80_100_b50_mb50_all_aug', 
	'deepmedic_ce_75_100_125_b50_mb50_all_aug', 'deepmedic_ce_22x18x6_all_aug', 'deepmedic_ce_75_100_125_b50_mb50_all', 
	'deepmedic_ce_28x20x12_all_aug', 'deepmedic_ce_45_60_75_b50_mb50_all', 'munet_dice_all', 'unet_dice_c25_all', 
	'unet_ce_hard_c25', 'unet_ce_hard_per_im_c25']
	}
}
#root_dir = '/media/hdd1/pkao/brats2018/validation'
#file_list = os.path.join(root_dir, 'test.txt')
#root_dir = '/media/hdd1/pkao/brats2018/training'
#file_list = os.path.join(root_dir, 'all.txt')
root_dir = '/usr/data/pkao/brats2018/testing'
file_list = os.path.join(root_dir, 'test.txt')

names = open(file_list).read().splitlines()

root = '/home/pkao/brats2017-master/output'

submission_name = 'ensemble_26_testing'

models = settings[submission_name]['models']

submission_dir = os.path.join('submissions', submission_name+'_uint8')

if not os.path.exists(submission_dir):
	os.makedirs(submission_dir)

for name in names:
	
	if 'HGG' in name or 'LGG' in name:
		name = name[4:]

	print(name)

	ari_dir = os.path.join(submission_dir, 'arith_dir')
	if not os.path.exists(ari_dir):
		os.makedirs(ari_dir)
	geo_dir = os.path.join(submission_dir, 'geo_dir')
	if not os.path.exists(geo_dir):
		os.makedirs(geo_dir)

	ari_oname = os.path.join(ari_dir, name+'.nii.gz')
	geo_oname = os.path.join(geo_dir, name+'.nii.gz')

	ari_preds = 0.0
	geo_preds = 0.0

	for k, model in enumerate(models):
		#fname = os.path.join(root, models[k], name+'_preds.npz')
		fname = os.path.join(root, models[k], 'test', name+'_preds.npz')
		prob_map = np.load(fname)
		prob_map_uint8 = prob_map['data']
		#print(prob_map_uint8.shape)
		prob_map_float32 = prob_map_uint8.astype(np.float32)/255.0
		#print(np.amax(prob_map_float32), np.amin(prob_map_float32))
		#print(prob_map_float32.dtype)
		ari_preds += prob_map_float32
		geo_preds += np.log(prob_map_float32+0.001)

	ari_preds = ari_preds.argmax(0).astype('uint8')
	geo_preds = geo_preds.argmax(0).astype('uint8')
	print(np.amax(ari_preds), np.amax(geo_preds))
	print(ari_preds.shape, geo_preds.shape)
	ari_img = nib.Nifti1Image(ari_preds, None)
	nib.save(ari_img, ari_oname)

	geo_img = nib.Nifti1Image(geo_preds, None)
	nib.save(geo_img, geo_oname)
