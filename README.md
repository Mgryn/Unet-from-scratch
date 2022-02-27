# Unet-from-scratch

The repository contains an implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

The file structure follows the one in [Recurrent Visual Attention Model implementation](https://github.com/Mgryn/recurrent-visual-attention) by kevinzakka, as it's very well organized and I have previous experience with this code.

## Motivation

I choose the U-net as it is easy to implement and very effective network in various image processing tasks. 

## File structure

- config.py: parameters of the network
- main.py: loads the dataset and starts training
- trainer.py: contains class for setting the model and starting training
- model.py: contains the U-net model
- modules.py: contains elements of the model
- data_utils.py: functions necessary for preprocessing, data loading and augmentations

## Future work

The project is not done yet, currently it consists of training an validation steps.
The test stage and tracking / plotting of the parameters needs to be implemented.

When the model will be finished, it would be good to introduce possbitity to work with bigger nuber of classes - currently it supports two.
