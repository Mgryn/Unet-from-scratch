
import os
import time
from tqdm import tqdm
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from model import Unet
import pdb

class Trainer:
    def __init__(self, config, dataloader):
        """Initialize the network with parameters from config file"""
        self.unet = Unet()
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.momentum = config.mn
        self.checkpoint_dir = config.checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config.train:
            self.train_loader = dataloader[0]
            self.valid_loader = dataloader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)

        else:
            self.test_loader = dataloader
            self.num_test = len(self.test_loader.dataset)

        self.unet.to(self.device)

        self.optimizer = optim.Adam(self.unet.parameters(), lr = self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_valid_acc = 0

    
    def train_one_epoch(self):

        self.unet.train()
        loss_sum = 0
        acc_sum = 0
        tic = time.time()
        num_x = 0

        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                out, probabilities = self.unet(x)
                loss = self.criterion(out, y.float())
                loss_sum += loss

                predictions = torch.max(probabilities, dim=1)[1]
                correct = (predictions == y[:, 0, :, :]).float()
                acc_sum += correct.sum()

                loss.backward()
                self.optimizer.step()

                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f}".format(
                            (toc - tic), loss.item()
                        )
                    )
                )
                pbar.update(self.batch_size)
        
        loss_val = loss_sum / self.num_train
        pixel_num = self.num_train * x.shape[2]*x.shape[3]
        acc_val = acc_sum / pixel_num
        return loss_val, acc_val

    @torch.no_grad()
    def validate(self):

        loss_sum = 0
        acc_sum = 0

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)
            out, probabilities = self.unet(x)
            loss = self.criterion(out, y.float())
            loss_sum += loss
            predictions = torch.max(probabilities, dim=1)[1]
            correct = (predictions == y[:, 0, :, :]).float()
            acc_sum += correct.sum()

        loss_val = loss_sum / self.num_valid
        pixel_num = self.num_valid * x.shape[2] * x.shape[3]
        acc_val = acc_sum / pixel_num

        return loss_val, acc_val


    def train(self):

        for epoch in range(self.epochs):

            self.running_loss = 0
            
            train_loss, train_acc = self.train_one_epoch()
            valid_loss, valid_acc = self.validate()

            self.scheduler.step(-valid_acc)

            is_best = train_loss > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.unet.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_loss": self.best_valid_acc,
                },
                is_best,
            )

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename =  "Unet_ckpt.pth.tar"
        ckpt_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = "Unet_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.checkpoint_dir, filename))
