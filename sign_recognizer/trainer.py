import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class LitGTSRBTrainer(pl.LightningModule):
    """
    Обёртка над GtsrbModel в стиле PyTorch Lightning.
    """

    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3):
        """
        :param model: GTSRBNet.
        :param learning_rate.
        """
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.save_hyperparameters("learning_rate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        """
        Создаём оптимизатор.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Шаг обучения.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Шаг теста.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return loss
