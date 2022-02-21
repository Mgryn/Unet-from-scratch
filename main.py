import torch
import torch.nn as nn
import torch.optim as optim
from model import Unet
from config import get_config
import tqdm

def load_data(directory):
    # TODO
    return []

def train_one_epoch():
    # TODO
    pass

def train(config):
    net = Unet()
    epochs = config.epochs
    batch_size = config.batch_size
    lr = config.lr
    momentum = config.mn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(net.parameters(), lr = lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = nn.CrossEntropyLoss()
    data = load_data()
    
    step = 0

    for epoch in range(epochs):
        net.train()
        loss = 0

        with tqdm(total=data.size()):
            train_one_epoch()

def main(config):

    torch.manual_seed(config.seed)
    #TODO: dataloader implementation
    data = load_data(config.data_dir)
    print(data)

    train(config)

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
