# KiU-Net-pytorch

 <a href="https://comingsoon"> Paper </a> |  <a href="https://sites.google.com/view/kiunet/home"> Project Page </a>


Official Pytorch Code for the paper "KiU-Net: Towards Accurate Segmentation of Biomedical Images using Over-complete Representations" , appearing in MICCAI 2020.

# Introduction

In a generic "encoder-decoder" architecture , the initial few blocks of the encoder learn low-level features of the data while the later blocks learn the high-level features. Eventually, the encoder learns to map the data to lower dimensionality (in the spatial sense). The increasing receptive field size over the depth of the network, constrains the network to focus more on the higher-level features. In our proposed architecture , we introduce Ki-Net where we use overcomplete representations which constraints the receptive field from increasing. This is done by a simple change in the architecture of encoder where max-pooling is replaced by up-sampling. This helps the filters in deep layers focus more on the low-level details helping in fine segmentation.  This network, when augmented with U-Net is termed as KiU-Net which results in significant improvements in the case of segmenting small anatomical landmarks and blurred noisy boundaries while obtaining better overall performance. More details can be found in the paper.

<p align="center">
  <img src="img/arch.png" width="300"/>
</p>

# Using the Code

- Clone this repository:
```bash
git clone https://github.com/jeya-maria-jose/KiU-Net-pytorch
cd KiU-Net-pytorch
```

### Prerequisites:

- Python 3.6
- Pytorch 1.4

<a href="https://pytorch.org/ "> Pytorch Installation </a>  

### Dataset Preparation

Prepare the dataset in the following format for easy use of the code. The train and test folders should contain two subfolders each: img and label. Make sure the images their corresponding segmentation masks are placed under these folders and have the same name for easy correspondance. Please change the data loaders to your need if you prefer not preparing the dataset in this format.


```bash
Train Folder-----
      img----
          0001.png
          0002.png
          .......
      label---
          0001.png
          0002.png
          .......
Val Folder-----
      img----
          0001.png
          0002.png
          .......
      label---
          0001.png
          0002.png
          .......
Test Folder-----
      img----
          0001.png
          0002.png
          .......
      label---
          0001.png
          0002.png
          .......

```

### Training Command:

<code> python train.py --train_dataset "enter train directory" --val_dataset "enter validation directory" --direc 'path for results to be saved' --batch_size 1 --epoch 400 --save_freq 10 --modelname "kiunet" --learning_rate 0.0001
 </code>

### Testing Command:

<code> python test.py --loaddirec "./saved_model_path/model_name.pth" --val_dataset "test dataset directory" --direc 'path for results to be saved' --batch_size 1 --modelname "kiunet"
 </code>


# Acknowledgement:

The dataloader code is taken from <a href="https://github.com/cosmic-cortex/pytorch-UNet"> pytorch-UNet </a>


# Citation:

Coming Soon

Open an issue in case of any queries.