import torch
import pytorch_lightning as pl
import typing
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence


class PlBaseModel(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module, loss_fn: typing.Callable, **kwargs):
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
        self.metrics = kwargs.pop("metrics", [])
        self.checkpoint = kwargs.pop('load_checkpoint', None)
        self.monitor = kwargs.pop("monitor", "val_loss")

        self.hparams.update({k: v for k, v in locals().items() if 'hp' in k})

        if self.checkpoint is not None:
            model=self.model.load_state_dict(self.checkpoint)
        self.loss_fn = loss_fn
        self.monitor_logs = {
            'train_loss': [],
            'val_loss': [],
            'val_best_loss': 10e9,
            'val_best_metric': 10e9
        }

        self.monitor_logs.update({f'{self.monitor}': []})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn( y_hat,y)
        self.monitor_logs.get("train_loss").append(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.monitor_logs.get("val_loss").append(val_loss)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.metrics is not None:
            metric = self.metrics(y_hat, y)
            self.monitor_logs.get(self.monitor).append(metric)

            return {'val_loss': val_loss, self.monitor: metric}
        else:
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
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': self.monitor}

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.monitor_logs.get('val_loss')) / len(self.monitor_logs.get('val_loss'))
        self.log('val_loss', avg_val_loss)
        self.monitor_logs.update({'val_loss': []})
        if self.metrics is not None and self.monitor != 'val_loss':
            avg_metrics = sum(self.monitor_logs.get(self.monitor)) / len(self.monitor_logs.get(self.monitor))
            self.log(self.monitor, avg_metrics)
            self.monitor_logs.update({self.monitor: []})
