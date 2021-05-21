import os
import time
import random
import sys
sys.path.insert(0,'/data1/inqlee0704/DL_code')

from model_util import get_model
from train_util import *
from engine import Segmentor
from dataloader import prep_dataloader


import numpy as np
import pandas as pd
from torch import nn
from torch.cuda import amp
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn import metrics, model_selection
from medpy.io import load
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class CFG:
    def __init__(self):
        self.name = "Recursive_UNet_CE_4downs"
        self.model = 'RecursiveUNet' # ['UNet', 'RecursiveUNet']
        self.optimizer = 'adam' # ['adam', 'adamp']
        self.scheduler = 'CosineAnnealingWarmRestarts' 
        # ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'ReduceLROnPlateau']
        self.epochs = 40 
        self.T_0 = self.epochs
        self.root_path = r"/data4/inqlee0704"
        self.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
        self.lr = 0.0001
        self.min_lr = 1e-6
        self.train_bs = 16
        self.valid_bs = 32
        self.test_results_dir = "RESULTS"
        self.mask = 'airway'
        self.save = True

if __name__ == "__main__": 
    seed_everything()
    start = time.time()

    c = CFG()
    n_case=256
    print('***********************************************************')
    print('Configuration: ')
    print(c.__dict__)
    print(f'n_case: {n_case}')
    print('***********************************************************')
    # Data
    train_loader, valid_loader = prep_dataloader(c,n_case=256)
    # Model
    model = get_model(c)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    # loss
    loss_function = nn.CrossEntropyLoss()
    # optimizer
    optimizer = get_optimizer(model,c)
    # scheduler
    scheduler = get_scheduler(optimizer,c)

    scaler = amp.GradScaler()
    eng = Segmentor(model=model, 
                 optimizer=optimizer,
                 scheduler=scheduler,
                 loss_fn=loss_function,
                 device=DEVICE,
                 scaler=scaler)

    if c.save:
        dirname = f'{c.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join('RESULTS',dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "model.pth")

    best_loss = np.inf
    best_dice = 0
    # Train
    for epoch in range(c.epochs):
        trn_loss, trn_dice = eng.train(train_loader)
        val_loss, val_dice = eng.evaluate(valid_loader)
        scheduler.step(val_loss)
        eng.epoch += 1
        print(f'Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}')
        print(f'Epoch: {epoch}, train dice: {trn_dice:5f}, valid dice: {val_dice:5f}')
       #  if val_dice > best_dice:
       #      best_dice = val_dice
       #      print('Best Dice:', best_dice)
       #      if c.save:
       #          torch.save(model.state_dict(), path)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Best loss:' , best_loss)
            if c.save:
                torch.save(model.state_dict(), path)

    end = time.time()
    print('Elapsed time: ' + str(end-start))
