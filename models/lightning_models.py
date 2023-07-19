import torch
import pytorch_lightning as pl
from utils.prediction_loops import predict_tensor_patches
import numpy as np
from typing import Union, Any, Optional, Callable
import torchio as tio

class Unet3D(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        metrics: Callable=None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        positional: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Instantiate a 3D U-Net model.

        Args:
            model (torch.nn.Module): The segmentation model to use.
            loss (torch.nn.Module): The loss function to use.

        Methods:
            forward: Forward pass of the model.
            training_step: Training step of the model.
            validation_step: Validation step of the model.
            predict_step: Predict step of the model.
        """
        super().__init__()

        self.in_channels = kwargs.get("in_channels", 1)
        self.depths = kwargs.get("depths", [3, 3, 9, 3])
        self.embed_dim = kwargs.get("embed_dim", 32)
        self.channel_multipliers = kwargs.get("channel_multipliers", [1, 2, 4, 8])
        self.classes = kwargs.get("classes", 1)
        self.final_activation = kwargs.get("final_activation", torch.nn.Identity())
        self.positional = positional

        if self.positional:
            self.positional_embed_dim = kwargs.get("positional_embed_dim", 32)
            self.positional_channels = kwargs.get("positional_channels", 3)
            self.model = model(
                in_channels=self.in_channels,
                depths=self.depths,
                embed_dim=self.embed_dim,
                channel_multipliers=self.channel_multipliers,
                classes=self.classes,
                final_activation=self.final_activation,
                positional_embed_dim=self.positional_embed_dim,
                positional_channels=self.positional_channels,)

        else:    
            self.model = model(
                in_channels=self.in_channels,
                depths=self.depths,
                embed_dim=self.embed_dim,
                channel_multipliers=self.channel_multipliers,
                classes=self.classes,
                final_activation=self.final_activation,
            )

        self.loss = loss

        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = None

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

        self.training_steps = kwargs.get("training_steps", 1000)
        self.initial_lr = kwargs.get("initial_lr", 1e-3)
        self.patch_size = kwargs.get("patch_size", (128, 128, 128))
        self.strides = kwargs.get("strides", (64, 64, 64))
        self.padding = kwargs.get("padding", "same")

        self.beta = kwargs.get("beta", 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional:
            x, pos = x
            return self.model(x, pos)
        return self.model(x)

    def training_step(self, batch, batch_idx):

        if self.positional:
            x, y, pos = batch
            y_pred = self((x, pos))
        else:
            x, y = batch
            y_pred = self(x)

        if y.dim() > 4:
            y = y.squeeze(1)

        loss = self.loss(y, y_pred) * self.beta

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        if self.metrics is not None:

            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(f'train_{name}', metric(y_pred, y), prog_bar=True, on_step=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        volume_pred = self.predict_step(
            volume=x,
            patch_size=self.patch_size,
            strides=self.strides,
            padding=self.padding,
            unpad=True,
            verbose=False,
            positional=self.positional,
        )
        volume_pred = volume_pred.unsqueeze(0)

        if y.dim() > 4:
            y = y.squeeze(1)

        loss = self.loss(y, volume_pred) * self.beta

        self.log("val_loss", loss, prog_bar=True, logger=True)

        if self.metrics is not None:

            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(f'val_{name}', metric(volume_pred, y), prog_bar=True, logger=True)

    def predict_step(self, volume: torch.Tensor, **kwargs) -> torch.Tensor:
        """Prediction step of the model.

        Args:
            volume (torch.Tensor): The volume to predict. Should be a 5D tensor with dimensions
                (batch_size, channels, z, y, x).

        Returns:
            pred (torch.Tensor): The predicted volume.
        """
        volume = volume.squeeze().squeeze()

        return predict_tensor_patches(tensor=volume, model=self.model, **kwargs)

    def configure_optimizers(self):
        
        optimizer = self.optimizer(self.parameters(), lr=self.initial_lr)
        scheduler = self.lr_scheduler(optimizer, T_max=self.training_steps, eta_min=1e-6)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class UnetIO(Unet3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA].float(), batch['label'][tio.DATA].float()
    
    def training_step(self, batch, batch_idx):

        x, y = self.prepare_batch(batch)

        if y.dim() > 4:
            y = y.squeeze(1)

        y_pred = self(x)

        loss = self.loss(y, y_pred) * self.beta

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(f'train_{name}', metric(y_pred, y), prog_bar=True, on_step=True, logger=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y = self.prepare_batch(batch)

        if y.dim() > 4:
            y = y.squeeze(1)

        y_pred = self.predict_step(volume=x,
            patch_size=self.patch_size,
            strides=self.strides,
            padding=self.padding,
            unpad=True,
            verbose=False,
            positional=self.positional,)
        
        y_pred = y_pred.unsqueeze(0)

        loss = self.loss(y, y_pred) * self.beta

        self.log('val_loss', loss, prog_bar=True, logger=True)

        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(f'val_{name}', metric(y_pred, y), prog_bar=True, logger=True)