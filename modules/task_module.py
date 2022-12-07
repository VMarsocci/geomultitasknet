import torch
from torchmetrics import MeanMetric, JaccardIndex
import pytorch_lightning as pl

from .utils import spatiotemporal_batches


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criteria,
        optimizer,
        scheduler=None,
        uda = False,
        geo_data = False
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.uda = uda
        self.geo_data = geo_data


    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None

            self.train_metrics = JaccardIndex(
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.val_metrics = JaccardIndex(
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = JaccardIndex(
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.val_loss = MeanMetric()

    def forward(self, input_im, idx):
        outputs = {}
        if self.model.name in ("FDMUNet", "UNet", "ResUNet"):
            outputs["x1"], outputs["x2"], outputs["x5"], outputs["logits"] = self.model(input_im)
        elif self.model.name in ("ConcatGeoUNet", "GeoUNet"):
            outputs["x1"], outputs["x2"], outputs["x5"], outputs["logits"] = self.model(input_im, idx)
        elif self.model.name == "GeoTimeTaskUNet":
            outputs["logits"], outputs["x_coord"], outputs["x_time"] = self.model(input_im)   
        elif self.model.name == "MultiTaskUNet":
            outputs["logits"], outputs["x_coord"] = self.model(input_im)    
        return outputs

    def step(self, batch, stage = "train"):
        if stage == "train":
            idx, images, targets = batch["source"]
            if self.uda:
                idx_t, im_t, _ = batch["target"]
        elif stage == "val":
            idx, images, targets = batch

        # if self.model.name 
        outputs = self.forward(images, idx)

        seg_criterion = self.criteria["segmentation"]
        loss = seg_criterion(outputs["logits"], targets.long())

        if self.uda and (stage == "train"):
            target_outputs = self.forward(im_t, idx_t)

            if self.criteria["constraint_name"] == "style_loss":
                constr_criterion= self.criteria["constraint"]
                constr_weight = self.criteria["constraint_weight"]
                style_loss1 = constr_criterion(outputs["x1"], target_outputs["x1"])
                style_loss2 = constr_criterion(outputs["x2"], target_outputs["x2"])
                loss = loss + constr_weight*style_loss1 + constr_weight*style_loss2
            elif self.criteria["constraint_name"] == "coral":
                constr_criterion= self.criteria["constraint"]
                constr_weight = self.criteria["constraint_weight"]
                f_l = torch.reshape(torch.permute(outputs["x5"], (0, 2, 3, 1)), (-1, outputs["x5"].shape[-1]))
                f_un = torch.reshape(torch.permute(target_outputs["x5"], (0, 2, 3, 1)), (-1, target_outputs["x5"].shape[-1]))
                coral_loss = constr_criterion(f_l, f_un)
                loss = loss + constr_weight*coral_loss
            elif self.criteria["constraint_name"] == "multitask_strategy":
                constr_criterion= self.criteria["constraint"]
                constr_weight = self.criteria["constraint_weight"]
                coords_l, _, _, _, _, _ = spatiotemporal_batches(idx, self.geo_data, pos_enc_coords = True, circle_encoding = True)
                coords_un, _, _, _, _, _ = spatiotemporal_batches(idx_t, self.geo_data, pos_enc_coords = True, circle_encoding = True)
                coord_loss1 = constr_criterion(outputs["x_coord"], coords_l)
                coord_loss2 = constr_criterion(target_outputs["x_coord"], coords_un)
                loss = loss + constr_weight*coord_loss1 + constr_weight*coord_loss2

        with torch.no_grad():
            proba = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(proba, dim=1)
            preds = preds.flatten(start_dim=1)  # Change shapes and cast target to integer for metrics computation
            targets = targets.flatten(start_dim=1).type(torch.int32)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch, stage = "train")
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_step_end(self, step_output):
        loss, preds, targets = (
            step_output["loss"].mean(),
            step_output["preds"],
            step_output["targets"]
        )
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log(
                "train_loss",
                self.train_epoch_loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                rank_zero_only=True
                )
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch, stage = "val")
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step_end(self, step_output):
        loss, preds, targets = (
            step_output["loss"].mean(),
            step_output["preds"],
            step_output["targets"]
        )
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.log(
            "val_miou",
            self.val_epoch_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.val_loss.reset()
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch
        x1, x2, x5, logits = self.forward(images)
        proba = torch.softmax(logits, dim=1)
        out_batch = {}
        out_batch["preds"] =  torch.argmax(proba, dim=1)
        out_batch["img"] = images
        return out_batch

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
                "name": "Scheduler"
            }
            config = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config
        else: return self.optimizer       
