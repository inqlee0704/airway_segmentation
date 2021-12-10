from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from sklearn import metrics
import numpy as np

from torch import nn


def Dice3d(a, b):
    intersection = np.sum((a != 0) * (b != 0))
    volume = np.sum(a != 0) + np.sum(b != 0)
    if volume == 0:
        return -1
    return 2.0 * float(intersection) / float(volume)


def cal_dice(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )
    return loss.mean()


def cal_dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )
    return loss.mean()


def cal_loss(outputs, targets, bce_weight=0.5):
    BCE_fn = nn.BCEWithLogitsLoss()
    bce_loss = BCE_fn(outputs, targets)
    preds = torch.sigmoid(outputs)
    dice_loss = cal_dice_loss(preds, targets)
    loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)
    return loss, bce_loss, dice_loss


class Segmentor:
    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        scheduler=None,
        device=None,
        scaler=None,
        combined_loss=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.combined_loss = combined_loss
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_bce_loss += bce_loss.item()
                pbar.set_description(
                    f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                )
            return epoch_loss / iters, epoch_dice_loss / iters, epoch_bce_loss / iters

        else:
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                pbar.set_description(f"loss:{loss:.3f}")
            return epoch_loss / iters

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    epoch_dice_loss += dice_loss.item()
                    epoch_bce_loss += bce_loss.item()

                    pbar.set_description(
                        f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                    )
                return (
                    epoch_loss / iters,
                    epoch_dice_loss / iters,
                    epoch_bce_loss / iters,
                )

        else:
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()

                    pbar.set_description(f"loss:{loss:.3f}")
                return epoch_loss / iters

    def inference(self, img_volume):
        # img_volume: [512,512,Z]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros(img_volume.shape)
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                out = self.model(slice.to(DEVICE, dtype=torch.float))
                pred = torch.argmax(out, dim=1)
                pred = np.squeeze(pred.cpu().detach())
                pred_volume[:, :, i] = pred
            return pred_volume

    def inference_multiC(self, img_volume):
        # img_volume: [3,512,512,Z]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros(img_volume.shape[1:])
        with torch.no_grad():
            for i in range(img_volume.shape[-1]):
                slice = img_volume[:, :, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0)
                out = self.model(slice.to(DEVICE, dtype=torch.float))
                # pred = torch.argmax(out, dim=1)
                pred = torch.sigmoid(out)
                pred = np.squeeze(pred.cpu().detach())
                pred_volume[:, :, i] = pred
            return pred_volume


class Segmentor_Z:
    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        scheduler=None,
        device=None,
        scaler=None,
        combined_loss=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.combined_loss = combined_loss
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                z = batch["z"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs, z)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_bce_loss += bce_loss.item()
                pbar.set_description(
                    f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                )
            return epoch_loss / iters, epoch_dice_loss / iters, epoch_bce_loss / iters

        else:
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                z = batch["z"].to(self.device)
                targets = batch["seg"].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs, z)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                pbar.set_description(f"loss:{loss:.3f}")
            return epoch_loss / iters

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    z = batch["z"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs, z)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    epoch_dice_loss += dice_loss.item()
                    epoch_bce_loss += bce_loss.item()

                    pbar.set_description(
                        f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                    )
                return (
                    epoch_loss / iters,
                    epoch_dice_loss / iters,
                    epoch_bce_loss / iters,
                )

        else:
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    z = batch["z"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs, z)
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    pbar.set_description(f"loss:{loss:.3f}")
                return epoch_loss / iters

    def inference(self, img_volume):
        # img_volume: [512,512,Z]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros(img_volume.shape)
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                z = i / (img_volume.shape[2] + 1)
                z = np.floor(z * 10)
                z = torch.tensor(z, dtype=torch.int64)
                out = self.model(slice.to(DEVICE, dtype=torch.float), z.to(DEVICE))
                pred = torch.argmax(out, dim=1)
                pred = np.squeeze(pred.cpu().detach())
                pred_volume[:, :, i] = pred
            return pred_volume

