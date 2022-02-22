# Unet-from-scratch

The repository contains an implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

The file structure follows the one in [Recurrent Visual Attention Model implementation](https://github.com/Mgryn/recurrent-visual-attention) by kevinzakka, as it's very well organized and I have previous experience with this code.

## Motivation

I choose the U-net as it is easy to implement and very effective network in various image processing tasks. 

## File structure

- config.py contains parameters of the network
- main.py 
- model.py - contains the U-net model
- modules.py - contains elements of the model

## Future work

This project is still in progress. The data-loading, image processing and training script have to be completed. After that, the network will be tested, and the code will be commented.

Additional, refactoring of the network for example will be beneficial to be able to work with different image input size.