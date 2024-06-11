from importlib import import_module

import sys
from model import VisionTransformer
from dataset import ImageNetDataset
# from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
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
    dataset_size = len(train_dataset)
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,      #cpu 코어 절반
        drop_last=True)
    return train_loader, dataset_size

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(model, data_loader):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs+1
    all_losses = []
    all_accs = []
    for epoch in range(epochs):
        loss = 0
        loss_epoch = 0
        model.train()
        acc = 0.0
        correct = 0
        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            x_indices = torch.argmax(outputs, dim=1)
            y_indices = torch.argmax(y, dim=1)
            correct += (y_indices==x_indices).sum().item()

            loss_epoch += loss.item()
        loss_epoch /= len(data_loader)
        acc = correct / len_dataset         #change here later
        print(loss_epoch)

        all_losses.append(loss_epoch)
        all_accs.append(acc)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    model = VisionTransformer(dim=768, depth=12, heads=12, output_dim=1000, img_dim=[3,224,224], patch_dim=[3,16,16], batch_size=args.batch_size, mlp_dim=3072)
    dataset, len_dataset = make_dataset()
    print(f"Image len: {len_dataset}")

    train(model, dataset)