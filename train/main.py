import os
from dotenv import load_dotenv
import time
import random
import wandb

from UNet import RecursiveUNet
from engine import Segmentor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
from dataloader import prep_dataloader

import numpy as np
from torch import nn
from torch.cuda import amp
import torch
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def wandb_config():
    config = wandb.config
    # ENV
    config.data_path = os.getenv('VIDA_PATH')
    config.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
    config.test_results_dir = "RESULTS"
    config.name = 'UNet_baseline_relu'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.mask = 'airway'
    config.model = 'UNet'
    config.activation = 'relu'
    config.optimizer = 'adam'
    config.scheduler = 'CosineAnnealingWarmRestarts'

    config.learning_rate = 0.0001
    config.train_bs = 16
    config.valid_bs = 32

    config.save = True
    config.debug = False
    if config.debug:
        config.epochs = 1
    else:
        config.epochs = 30
    return config

if __name__ == "__main__": 
    load_dotenv()
    seed_everything()
    wandb.init(project='airway')
    config = wandb_config()
    
    # Data
    n_case = 64
    if config.debug: # only use 10 cases, 1 epoch
        train_loader, valid_loader = prep_dataloader(config,n_case=5)
    else:
        train_loader, valid_loader = prep_dataloader(config,n_case=n_case)

    # Model #
    # Activation
    if config.activation == 'relu':
        activation_layer = nn.ReLU(inplace=True)
    elif config.activation == 'elu':
        activation_layer = nn.ELU(inplace=True)
    elif config.activation == 'leakyrelu':
        activation_layer = nn.LeakyReLU(inplace=True)

    model = RecursiveUNet(activation=activation_layer)
    model.to(config.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=config.epochs, 
                                                T_mult=1, 
                                                eta_min=1e-8,
                                                last_epoch=-1)

    scaler = amp.GradScaler()
    eng = Segmentor(model=model, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_function,
                    device=config.device,
                    scaler=scaler)

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join('RESULTS',dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "airway_UNet.pth")

    best_loss = np.inf
    best_dice = 0
    # Train
    wandb.watch(eng.model,log='all',log_freq=10)
    for epoch in range(config.epochs):
        trn_loss, trn_dice = eng.train(train_loader)
        val_loss, val_dice = eng.evaluate(valid_loader)
        wandb.log({'epoch': epoch, 'trn_loss': trn_loss, 'val_loss': val_loss,
                    'trn_dice': trn_dice, 'val_dice': val_dice})
        if config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        eng.epoch += 1
        print(f'Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}')
        print(f'Epoch: {epoch}, train dice: {trn_dice:5f}, valid dice: {val_dice:5f}')
        #  if val_dice > best_dice:
        #      best_dice = val_dice
        #      print('Best Dice:', best_dice)
        #      if config.save:
        #          torch.save(model.state_dict(), path)
        if val_loss < best_loss:
            best_loss = val_loss
            best_dice = val_dice
            print(f'Best loss: {best_loss} with {best_dice}')
            if config.save:
                torch.save(model.state_dict(), path)
                wandb.save(path)
                