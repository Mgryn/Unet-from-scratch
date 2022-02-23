import cv2
import glob
import numpy as np
import albumentations as A
from PIL import Image, ImageSequence
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler


def preprocessing(augmentations=20):
    """Loads the data form multipage tiff files, adds transforms and saves
        desired number of augmentations for each page."""
    # Load the files
    file = Image.open('data/train-volume.tif')
    images = [np.array(page) for page in ImageSequence.Iterator(file)]
    file = Image.open('data/train-labels.tif')
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
        for j in range(augmentations):
            index = i*augmentations + j
            transformed = transform(image = img, mask=lab)
            filename = 'data/train_img_%03d.png' % index
            print(filename)
            cv2.imwrite(filename, transformed['image'])
            filename = 'data/train_lab_%03d.png' % index
            cv2.imwrite(filename, transformed['mask'])


def normalize(images, label=False):
    images = np.array(images, dtype=float)
    images /= 255.0
    if not label:
        images = 2*images -1
    # transpose the array so it fits to network inputs
    images = np.transpose(images, (0, 3, 1, 2))
    return images

def load_data(data_dir):
    img_files = sorted(glob.glob(data_dir+'/train_img*.png'))
    label_files = sorted(glob.glob(data_dir+'/train_lab*.png'))
    images = [cv2.imread(file) for file in img_files]
    labels = [cv2.imread(file) for file in label_files]
    # normalize images to <-1;1>
    images = normalize(np.array(images))
    # normalize data to 1 or 0 values (membrane or not)
    labels = normalize(labels, label=True)
    return images, labels

def train_loader(
    data_dir, 
    batch_size=1, 
    valid_size=0.1,
    random_seed=66,
    num_workers=1):
    """Creates a dataloader for a training set."""

    images, labels = load_data(data_dir)
    train_size = images.shape[0]
    print('train sizeL: ', train_size)
    indices = list(range(train_size))
    split = int(np.floor(valid_size * train_size))
    
    #shuffling images and labels
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataset = TensorDataset(images, labels)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader

def test_loader(
    data_dir, 
    batch_size=1, 
    random_seed=66,
    num_workers=1):
    """Creates a dataloader for a training set."""

    images, labels = load_data(data_dir)
    dataset = TensorDataset(images, labels)

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader