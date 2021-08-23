import os
from dotenv import load_dotenv
import time
import random
import wandb

from UNet import RecursiveUNet
import segmentation_models_pytorch as smp

from engine import Segmentor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
from dataloader import prep_dataloader

import numpy as np
from torch import nn
from torch.cuda import amp
import torch
from torchsummary import summary

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
    config.name = 'UNet_resnet'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.mask = 'airway'
    config.model = 'UNet'
    # config.encoder = 'timm-efficientnet-b5'
    config.activation = 'relu'
    config.optimizer = 'adam'
    config.scheduler = 'CosineAnnealingWarmRestarts'
    config.loss = 'BCE+dice'
    config.bce_weight = 0.5
    # config.pos_weight = 1

    config.learning_rate = 0.0001
    config.train_bs = 4
    config.valid_bs = 8
    config.aug = True

    config.save = False
    config.debug = True 
    if config.debug:
        config.epochs = 1
        config.project = 'debug'
    else:
        config.epochs = 30
        config.project = 'airway'
    return config

if __name__ == "__main__": 
    load_dotenv()
    seed_everything()
    config = wandb_config()
    wandb.init(project=config.project)
    
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

    # loss 
    # if config.loss == 'BCE':
    #     pos_weight=torch.ones([1])*config.pos_weight
    #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.device))
    # else:
    #     loss_fn = nn.CrossEntropyLoss()

    model = RecursiveUNet(num_classes=1,in_channels=2, activation=activation_layer)
    # model = smp.Unet(config.encoder, in_channels=1)
    # model = smp.FPN(config.encoder, in_channels=1)

    model.to(config.device)
    summary(model,(2,512,512))

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
                    # loss_fn=loss_fn,
                    device=config.device,
                    scaler=scaler)

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join('RESULTS',dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "airway_UNet.pth")

    best_loss = np.inf
    # Train
    wandb.watch(eng.model,log='all',log_freq=10)
    for epoch in range(config.epochs):
        trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
        val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
        wandb.log({'epoch': epoch,
         'trn_loss': trn_loss, 'trn_dice_loss': trn_dice_loss, 'trn_bce_loss': trn_bce_loss,
         'val_loss': val_loss, 'val_dice_loss': val_dice_loss, 'val_bce_loss': val_bce_loss})

        if config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        eng.epoch += 1
        print(f'Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print(f'Best loss: {best_loss} at Epoch: {eng.epoch}')
            if config.save:
                torch.save(model.state_dict(), path)
                wandb.save(path)
