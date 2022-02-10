Install custom pytorch kernels from https://github.com/thuyen/multicrop

## Required Python libraries

nibabel, nipype, natsort, SimpleITK

`pip install nibabel,nipype,natsort,SimpleITK`

## Required Sofware

FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

## How to run

### Change:

```
experiments/settings.yaml
```
to point to data directories. These are general settings, applied to all experiments. 

### Preprocess data (look at the script for more details):

```
python prep.py
```

### For 3D KiU-Net run:

```
python train_kiunet.py --gpu 0 --cfg kiunet3d_n8
```

### Prediction

To make predictions, run `predict_kiunet.py` with similar arguments

### Model File

Kiunet3d models can be found in "models/unet.py"

### Testing

Once you get the predictions, upload them in the [CBICA portal](https://ipp.cbica.upenn.edu/) to get the performance metrics.

### Acknowledgement.

This code is built on top of the awesome dataloaders provided by [Thuyen Ngo](https://github.com/thuyen), found at [Link](https://github.com/ieee820/BraTS2018-tumor-segmentation).
