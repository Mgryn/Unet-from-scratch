import torch
import torch.nn as nn
import torch.optim as optim
from model import Unet

class Trainer:
    def __init__(self, config):
        """Initialize the network with parameters from config file"""
        self.net = Unet()
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.momentum = config.mn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr, momentum=self.momentum)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        self.criterion = nn.CrossEntropyLoss()

        ## TO BE CONTINUED
    
    
    def train_one_epoch(self):
        # TODO
        pass

    def train(self):
        
        # not yet impelmented
        return
        step = 0

        for epoch in range(self.epochs):
            self.net.train()
            loss = 0

            # with tqdm(total=data.size()):
            #     train_one_epoch()
            