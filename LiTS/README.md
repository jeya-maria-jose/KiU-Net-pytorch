# Liver segmengtation using KiU-Net:

This code is built on top of [MICCAI-Lits17](https://github.com/assassint2017/MICCAI-LITS2017)

## Requirements:

```bash
numpy==1.14.2
torch==1.0.1.post2
visdom==0.1.8.8
pandas==0.23.3
scipy==1.0.0
tqdm==4.40.2
scikit-image==0.13.1
SimpleITK==1.0.1
pydensecrf==1.0rc3
```

## Dataset

Liver tumor Segmentation Challenge (LiTS) contain 131 contrast-enhanced CT images provided by hospital around the world. 3DIRCADb dataset is a subset of LiTS dataset with case number from 27 to 48. we train our model with 111 cases from LiTS after removeing the data from 3DIRCADb and evaluate on 3DIRCADb dataset. For more details about the dataset: https://competitions.codalab.org/competitions/17094

## Running the code

Check parameter.py to change the data directory and other settings. The model files can be found in in "net/models.py". Use train.py to train the network. Use val.py to get the predictions and performance metrics.
  
