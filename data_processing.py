
from PIL import Image, ImageSequence
import numpy as np
import albumentations as A
import cv2

# Load the files
file = Image.open('unet-master/data/membrane/train-volume.tif')
images = [np.array(page) for page in ImageSequence.Iterator(file)]
file = Image.open('unet-master/data/membrane/train-labels.tif')
labels = [np.array(page) for page in ImageSequence.Iterator(file)]

# images size (knowing it is square)
size = images[0].shape[0]
# set of augumentations for the images
transform = A.Compose([
    A.ShiftScaleRotate(p=0.3),
    A.Flip(p=0.3),
    A.RandomSizedCrop((size-70, size), size, size, p=0.3),
    A.ElasticTransform(interpolation=cv2.INTER_CUBIC, p=1)
]) 

# Creation of 20 augumentations from each image
for i, (img, lab) in enumerate(zip(images, labels)):
    for j in range(20):
        index = i*20 + j
        transformed = transform(image = img, mask=lab)
        filename = 'data/train_img_%03d.png' % index
        print(filename)
        cv2.imwrite(filename, transformed['image'])
        filename = 'data/train_lab_%03d.png' % index
        cv2.imwrite(filename, transformed['mask'])
