import torch
import data_utils
from config import get_config
from trainer import Trainer


def main(config):

    if config.preprocessing:
        data_utils.preprocessing()
    
    torch.manual_seed(config.seed)

    if config.train:
        dataset = data_utils.train_loader(
            config.data_dir,
            config.batch_size,
            config.valid_size,
            config.seed,
        )
    else:
        dataset = data_utils.test_loader(
            config.data_dir,
            config.batch_size,
            config.seed,
        )
        
        Trainer(config, dataset)


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
