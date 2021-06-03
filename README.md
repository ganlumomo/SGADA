# Self-training Guided Adversarial Domain Adaptation For Thermal Imagery

<p align="center">
  <img src="images/activation_maps.png" width="400">
</p>

If you make use of this code, please cite the following paper:
```
@inproceedings{sgada2021,
  title={Self-training Guided Adversarial Domain Adaptation For Thermal Imagery},
  author={Akkaya, Ibrahim Batuhan and Altinel, Fazil and Halici, Ugur},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2021}
}
```

## Overview
This repository contains official implementation of "[Self-training Guided Adversarial Domain Adaptation For Thermal Imagery](https://arxiv.org/abs/1801.07939)" paper (accepted to CVPR 2021 [Perception Beyond the Visible Spectrum (PBVS)](https://pbvs-workshop.github.io/) workshop).

![](/images/sgada.png)

## Environment
- Python 3.8.5
- PyTorch 1.6.0

To install the environment using Conda:
```
$ conda env create -f requirements_conda.yml
```

This command creates a Conda environment named `sgada`. The environment includes all necessary packages for training of SGADA method. After installation of the environment, activate it using the command below:
```
$ conda activate sgada
```

### Note
Before running the training code, make sure that `DATASETDIR` environment variable is set to your dataset directory.

## Folder Structure For Dataset
Prepare your dataset folder as shown in the structure below.
```
DATASET_DIR
└── sgada_data
    ├── flir
    │   ├── train
    │   │   ├── bicycle
    │   │   ├── car
    │   │   └── person
    │   ├── val
    │   │   ├── bicycle
    │   │   ├── car
    │   │   └── person
    │   ├── test_wconf_wdomain_weights.txt
    │   └── validation_wconf_wdomain_weights.txt
    └── mscoco
       ├── train
       │   ├── bicycle
       │   ├── car
       │   └── person
       └── val
           ├── bicycle
           ├── car
           └── person
```

`test_wconf_wdomain_weights.txt` and `validation_wconf_wdomain_weights.txt` files can be found [here](/files). These files have the fields below. 
```
filePath, classifierPrediction, classifierConfidence, discriminatorPrediction, discriminatorConfidence, sampleWeight
```
If you want to generate pseudo-labelling files by yourself, your pseudo-labelling files should follow the given order. In order to obtain confidences and predictions, you can follow the training scheme in https://github.com/fazilaltinel/ADDA.PyTorch-resnet.
## Acknowledgement
This repo is mostly based on:
- https://github.com/Fujiki-Nakamura/ADDA.PyTorch
- https://github.com/fazilaltinel/ADDA.PyTorch-resnet
