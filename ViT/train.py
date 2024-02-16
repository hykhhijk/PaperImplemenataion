from importlib import import_module

import sys
from model import VisionTransformer
from dataset import ImageNetDataset
# from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import augmentation
import argparse

import random
import numpy as np
# import wandb
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#check path
# print(sys.path)

def make_dataset():
    train_transform, val_transform = augmentation.get_transform(224)
    train_dataset = ImageNetDataset(transforms= train_transform)
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,      #cpu 코어 절반
        drop_last=True)
    return train_loader

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(model, data_loader):
    epochs = args.epochs+1
    all_losses = []
    all_accs = []
    for epoch in range(epochs):
        loss = 0
        loss_epoch = 0
        model.train()
        acc = 0.0
        correct = 0
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            indices = torch.argmax(outputs, dim=1)
            correct += (y==indices).sum().item()

            loss_epoch += loss.item()
        loss_epoch /= len(data_loader)
        acc = correct / 4500
        print(loss_epoch)

        all_losses.append(loss_epoch)
        all_accs.append(acc)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()