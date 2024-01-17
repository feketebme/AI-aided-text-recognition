import torch
import pytorch_lightning as pl
import typing

class PlBaseModel(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module, loss:typing.Callable, **kwargs):
        super(PlBaseModel, self).__init__()
        self.scheduler = None
        self.optimizer = None
        self.model = backbone
        # Hp params
        self.hp_initial_lr_rate = kwargs.pop('hp_initial_lr_rate', 1e-3)
        self.hp_lr_scheduler_relative_threshold = kwargs.pop('hp_lr_scheduler_relative_threshold', 0.0)
        self.hp_lr_scheduler_patience = kwargs.pop("hp_lr_scheduler_patience", 10)
        self.hp_lr_reduce_factor = kwargs.pop("hp_lr_reduce_factor", .1)
        self.hp_weight_decay = kwargs.pop("hp_weight_decay", .01)
        self.hp_batch_size = kwargs.pop("hp_batch_size", None)

        self.hparams.update({k: v for k, v in locals().items() if 'hp' in k})
        self.checkpoint = kwargs.pop('load_checkpoint', None)
        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['state_dict'])
        self.loss_fn=loss


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        return val_loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hp_initial_lr_rate,
                                          weight_decay=self.hp_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.hp_lr_reduce_factor,
            patience=self.hp_lr_scheduler_patience,
            verbose=True,
            threshold=self.hp_lr_scheduler_relative_threshold,
            threshold_mode="rel"
        )
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': 'val_loss'}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_val_loss)
