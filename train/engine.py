from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from sklearn import metrics
import numpy as np


def Dice3d(a,b):
    intersection =  np.sum((a!=0)*(b!=0))
    volume = np.sum(a!=0) + np.sum(b!=0)
    if volume == 0:
        return -1
    return 2.*float(intersection)/float(volume)

class Segmentor:
    def __init__(self,model,optimizer,scheduler,loss_fn,device,scaler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        skip = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        for step, batch in pbar:
            self.optimizer.zero_grad()
            inputs = batch['image'].to(self.device,dtype=torch.float)
            # if BCEwithLogitsLoss,
            targets = batch['seg'].to(self.device, dtype=torch.float)
            # if CrossEntropyLoss,
            # targets = batch['seg'].to(self.device)
            with amp.autocast():
                outputs = self.model(inputs)
                # loss = self.loss_fn(outputs, targets[:, 0, :, :])
                loss = self.loss_fn(outputs, targets)
            preds = np.argmax(outputs.cpu().detach().numpy(),axis=1)
            targets = targets.cpu().detach().numpy()
            targets = np.squeeze(targets,axis=1)
            dice = Dice3d(preds,targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(self.epoch+step/iters)
            epoch_loss += loss.item()
            if dice==-1:
                skip += 1
            else:
                epoch_dice += dice
            pbar.set_description(f'loss:{loss:.2f}, dice:{dice:.4f}') 
        return epoch_loss/iters, epoch_dice/(iters-skip)

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        skip = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader),total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch['image'].to(self.device,dtype=torch.float)
                # if BCEwithLogitsLoss,
                targets = batch['seg'].to(self.device, dtype=torch.float)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                epoch_loss += loss.item()
                preds = np.argmax(outputs.cpu().detach().numpy(),axis=1)
                targets = targets.cpu().detach().numpy()
                targets = np.squeeze(targets,axis=1)
                dice = Dice3d(preds,targets)
                if dice==-1:
                    skip += 1
                else:
                    epoch_dice += dice
                pbar.set_description(f'loss:{loss:.2f}, dice:{dice:.4f}') 
            return epoch_loss/iters, epoch_dice/(iters-skip)
