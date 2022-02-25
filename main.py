import torch
import data_utils
from config import get_config
from trainer import Trainer

def main(config):

    if config.preprocessing:
        data_utils.preprocessing(config, config.augmentations)
    
    torch.manual_seed(config.seed)

    if config.train:
        data_dir = config.data_dir + 'train/'
        dataset = data_utils.train_loader(
            data_dir,
            config.batch_size,
            config.valid_size,
            config.seed,
        )
    else:
        data_dir = config.data_dir + 'test/'
        dataset = data_utils.test_loader(
            data_dir,
            config.batch_size,
        )    
    
    Trainer(config, dataset)


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
