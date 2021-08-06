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


class CFG:
    
    def __init__(self):
        self.data_path = os.getenv('VIDA_PATH')
        self.name = "UNet_baseline"
        self.model = 'RecursiveUNet' # ['UNet', 'RecursiveUNet']
        self.optimizer = 'adam' # ['adam', 'adamp']
        self.scheduler = 'CosineAnnealingWarmRestarts' 
        # ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'ReduceLROnPlateau']
        self.epochs = 10
        self.T_0 = self.epochs

        self.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
        self.lr = 0.0001
        self.min_lr = 1e-6
        self.train_bs = 16
        self.valid_bs = 32
        self.test_results_dir = "RESULTS"
        self.mask = 'airway'
        self.save = False
        self.debug = False


def tune_hyper():
    
    with wandb.init() as run:
        print('config:', dict(run.config))
        # Model
        if run.config.activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        model = RecursiveUNet(activation=activation)
        # model = RecursiveUNet(activation=nn.ELU(inplace=True))
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0=c.T_0, 
                                                    T_mult=1, 
                                                    eta_min=1e-6,
                                                    last_epoch=-1)

        scaler = amp.GradScaler()
        eng = Segmentor(model=model, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_function,
                    device=DEVICE,
                    scaler=scaler)


        best_loss = np.inf
        best_dice = 0
        # Train
        wandb.watch(eng.model,log='all',log_freq=10)
        for epoch in range(c.epochs):
            trn_loss, trn_dice = eng.train(train_loader)
            val_loss, val_dice = eng.evaluate(valid_loader)
            wandb.log({'epoch': epoch, 'trn_loss': trn_loss, 'val_loss': val_loss,
                        'trn_dice': trn_dice, 'val_dice': val_dice})
            if c.scheduler == 'ReduceLROnPlateau':
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
                best_dice = val_dice
                print(f'Best loss: {best_loss} with {best_dice}')
        torch.cuda.empty_cache()

if __name__ == "__main__": 
    load_dotenv()
    seed_everything()
    # wandb.init(project='airway')

    sweep_config = {
        'name':'UNet',
        'method':'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'activation': {
                'values' : ['elu','relu']
            }
        }
        # 'parameters': {
        #     'epochs': {
        #         'values' : [10,20]
        #     },
        #     'learning_rate' : {
        #         'min' : 0.0001,
        #         'max' : 0.01
        #     }
        # }

    }
    sweep_id = wandb.sweep(sweep_config,project='airway')

    c = CFG()
    n_case = 64
    print('***********************************************************')
    print('Configuration: ')
    print(c.__dict__)
    print('***********************************************************')
    
    # Data
    if c.debug: # only use 10 cases, 1 epoch
        train_loader, valid_loader = prep_dataloader(c,n_case=5)
        c.epochs = 2
    else:
        train_loader, valid_loader = prep_dataloader(c,n_case=n_case)

    count = 2
    wandb.agent(sweep_id, function=tune_hyper, count=count)