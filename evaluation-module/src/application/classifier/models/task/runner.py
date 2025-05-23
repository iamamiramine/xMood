import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from application.classifier.models.metric import accuracy, kl_divergence


class Runner(LightningModule):
    def __init__(self, model, config, eval_type):
        super().__init__()
        self.model = model
        self.cls_type = getattr(model, "cls_type", "AV")

        # Use KL divergence loss for mood vectors, CrossEntropy for valence-arousal
        if self.cls_type == "MOOD":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
            self.use_log_softmax = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.use_log_softmax = False

        self.config = config
        self.eval_type = eval_type

        # For saving validation outputs
        self.validation_step_outputs = []

    def forward(self, x):
        output = self.model(x)
        # Apply log_softmax for KL divergence if needed
        if self.use_log_softmax:
            output = torch.log(torch.clamp(output, min=1e-8, max=1.0))
        return output

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer=opt, T_0=self.config.T_0)
        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
            "reduce_on_plateau": False,  # For ReduceLROnPlateau scheduler
            "monitor": "val_loss",  # Metric to monitor
        }
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)

        # Handle mood vectors vs. valence-arousal labels
        if self.cls_type == "MOOD":
            loss = self.criterion(prediction, label)
            # For logging, get both accuracy and KL div metrics
            acc = accuracy(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            kl_div = kl_divergence(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            self.log_dict(
                {
                    "train_loss": loss,
                    "train_acc": acc,
                    "train_kl_div": kl_div,
                },
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        else:
            # Original valence-arousal code
            loss = self.criterion(prediction, label.long())
            acc = accuracy(prediction, label.long())
            self.log_dict(
                {
                    "train_loss": loss,
                    "train_acc": acc,
                },
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        print(loss, flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)

        # Handle mood vectors vs. valence-arousal labels
        if self.cls_type == "MOOD":
            loss = self.criterion(prediction, label)
            acc = accuracy(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            kl_div = kl_divergence(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc, "val_kl_div": kl_div})
            return {"val_loss": loss, "val_acc": acc, "val_kl_div": kl_div}
        else:
            # Original valence-arousal code
            loss = self.criterion(prediction, label.long())
            acc = accuracy(prediction, label.long())
            self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc})
            return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_start(self):
        # Clear the outputs list at the start of each validation epoch
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = torch.mean(torch.stack([output["val_acc"] for output in outputs]))

        if self.cls_type == "MOOD":
            val_kl_div = torch.mean(torch.stack([output["val_kl_div"] for output in outputs]))
            self.log_dict(
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_kl_div": val_kl_div,
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            # Original valence-arousal code
            self.log_dict(
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Clear the outputs list after processing
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        audio, label, fname = batch
        prediction = self.forward(audio)

        # Handle mood vectors vs. valence-arousal labels
        if self.cls_type == "MOOD":
            loss = self.criterion(prediction, label)
            acc = accuracy(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            kl_div = kl_divergence(torch.exp(prediction) if self.use_log_softmax else prediction, label)
            return {"val_loss": loss, "val_acc": acc, "val_kl_div": kl_div, "prediction": prediction, "label": label, "fname": fname}
        else:
            # Original valence-arousal code
            loss = self.criterion(prediction, label.long())
            acc = accuracy(prediction, label.long())
            return {"val_loss": loss, "val_acc": acc, "prediction": prediction, "label": label, "fname": fname}

    def on_test_epoch_start(self):
        # Initialize a list to collect test step outputs
        self.test_step_outputs = []

    def test_step_end(self, batch_parts):
        # Collect outputs in the list
        self.test_step_outputs.append(batch_parts)
        return batch_parts

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = torch.mean(torch.stack([output["val_acc"] for output in outputs]))

        metrics_dict = {"test_loss": float(val_loss.detach().cpu()), "test_acc": float(val_acc.detach().cpu())}

        if self.cls_type == "MOOD":
            val_kl_div = torch.mean(torch.stack([output["val_kl_div"] for output in outputs]))
            metrics_dict["test_kl_div"] = float(val_kl_div.detach().cpu())

        if self.eval_type == "last":
            log_dict = {
                "last_loss": val_loss,
                "last_acc": val_acc,
            }
            if self.cls_type == "MOOD":
                log_dict["last_kl_div"] = val_kl_div

            self.log_dict(
                log_dict,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif self.eval_type == "best":
            log_dict = {
                "best_loss": val_loss,
                "best_acc": val_acc,
            }
            if self.cls_type == "MOOD":
                log_dict["best_kl_div"] = val_kl_div

            self.log_dict(
                log_dict,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.test_results = metrics_dict
        # predictions = [output["prediction"] for output in outputs]
        # labels = [output["label"] for output in outputs]
        # fnames = [output["fname"] for output in outputs]
        # pred = {"predictions": predictions,"labels": labels, "fnames": fnames}
        # torch.save(pred, "predictions.pt")

        # Clear the outputs list after processing
        self.test_step_outputs = []

        return metrics_dict
