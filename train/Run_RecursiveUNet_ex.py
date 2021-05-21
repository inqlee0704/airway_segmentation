import sys
sys.path.insert(0,'/home/inqlee0704/src/DL')
import numpy as np
from medpy.io import load
import pandas as pd
import os
from models import RecursiveUNet
from engine import Engine
from torch import nn
from torch.cuda import amp
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import time
from sklearn import metrics, model_selection
from torch.utils.tensorboard import SummaryWriter
from dataloader import prep_dataloader
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
        self.root_path = r"/data4/inqlee0704"
        self.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
        self.epochs = 10 
        self.lr = 0.0002
        self.train_bs = 8
        self.valid_bs = 8
        self.test_results_dir = "RESULTS"
        self.subjlist = pd.read_csv(os.path.join(self.root_path,self.in_file),sep='\t')    
        self.mask = 'airway'
        self.save = False
        
if __name__ == "__main__": 
    seed_everything()
    start = time.time()

    c = CFG()
    # Data
    train_loader, valid_loader = prep_dataloader(c,n_case=16)
    # Model
    model = RecursiveUNet(num_downs=4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    # loss
    loss_function = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=c.lr)
    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    scaler = amp.GradScaler()
    eng = Engine(model=model, 
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
    # Train
    for epoch in range(c.epochs):
        trn_loss = eng.train(train_loader)
        val_loss = eng.evaluate(valid_loader)
        scheduler.step(val_loss)
        eng.epoch += 1
        print(f'{epoch}, {trn_loss}, {val_loss}')
        if val_loss < best_loss:
            best_loss = val_loss
            print('Best loss:' , best_loss)
            if c.save:
                torch.save(model.state_dict(), path)

    end = time.time()
    print('Elapsed time: ' + str(end-start))
