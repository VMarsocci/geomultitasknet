import torch
from torchmetrics import MeanMetric, JaccardIndex, ConfusionMatrix
import pytorch_lightning as pl
import torchvision
import io
from PIL import Image

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from prettytable import PrettyTable

from .utils import spatiotemporal_batches, calc_miou


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criteria,
        optimizer,
        scheduler=None,
        uda = False,
        geo_data = False,
        metadata = False,
        config_name = False
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.uda = uda
        self.geo_data = geo_data
        self.metadata = metadata
        self.config_name = config_name


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

        elif stage == "test":
            self.cm_test_metrics = ConfusionMatrix(task="multiclass", 
                                                num_classes=self.num_classes)
            self.test_metrics = JaccardIndex(
                    num_classes=self.num_classes,
                    reduction = "none")

    def forward(self, input_im, idx):
        outputs = {}
        if self.model.name in ("FDMUNet", "UNet", "ResUNet18", "ResUNet34", "ResUNet50", "ResUNet101", "ResUNet152"):
            outputs["x1"], outputs["x2"], outputs["x5"], _, outputs["logits"] = self.model(input_im)
        elif self.model.name in ("ConcatGeoUNet", "GeoUNet"):
            outputs["x1"], outputs["x2"], outputs["x5"], outputs["logits"] = self.model(input_im, idx)
        elif self.model.name == "GeoTimeMultiTaskNet":
            outputs["logits"], outputs["x_coord"], outputs["x_time"] = self.model(input_im)   
        elif self.model.name == "GeoMultiTaskNet":
            outputs["logits"], outputs["x_coord"] = self.model(input_im)    
        return outputs

    def step(self, batch, stage = "train"):
        if stage == "train":
            idx, images, targets = batch["source"]
            if self.uda:
                idx_t, im_t, _ = batch["target"]
        elif stage == "val":
            idx, images, targets = batch

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
                coords_l, month_l, hour_l, _, _, _ = spatiotemporal_batches(idx, self.geo_data, 
                                                                            pos_enc_coords = self.metadata["pos_enc_coords"], 
                                                                            circle_encoding = self.metadata["circle_encoding"],
                                                                            encoding_freq= self.metadata["encoding_freq"],
                                                                            geo_noise = self.metadata["geo_noise"])
                coords_un, month_un, hour_un, _, _, _ = spatiotemporal_batches(idx_t, self.geo_data,                                                                             pos_enc_coords = self.metadata["pos_enc_coords"], 
                                                                            circle_encoding = self.metadata["circle_encoding"],
                                                                            encoding_freq= self.metadata["encoding_freq"],
                                                                            geo_noise = self.metadata["geo_noise"])
                coord_loss1 = constr_criterion(outputs["x_coord"], coords_l)
                coord_loss2 = constr_criterion(target_outputs["x_coord"], coords_un)
                if self.criteria["mt_time"]:
                    time_l = torch.cat([month_l, hour_l], dim = -1)
                    time_un = torch.cat([month_un, hour_un], dim = -1)
                    time_loss1 = constr_criterion(outputs["x_time"], time_l)
                    time_loss2 = constr_criterion(target_outputs["x_time"], time_un)
                    loss += coord_loss1 + coord_loss2 + constr_weight*time_loss1 + constr_weight*time_loss2
                else:
                    loss += constr_weight*coord_loss1 + constr_weight*coord_loss2

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _, preds, targets = self.step(batch, stage = "val")
        self.test_metrics(preds, targets)
        self.cm_test_metrics(preds, targets)

        self.log(
            "test_miou",
            self.test_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)

    def test_epoch_end(self, outputs):
        cm = self.cm_test_metrics.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(36,15)) 
        ax.set(xlabel='Predicted', ylabel='Actual')
        df_cm = pd.DataFrame(cm, range(self.num_classes), range(self.num_classes)).astype(np.int64)
        res = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues',  fmt=',d', ax = ax, cbar = False)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 15)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 15)
        plt.savefig(f"experiments/{self.config_name}/conf_matrix.jpeg")

        classes = ['autres', 'batiment', 'zone-permeable', 'zone-impermeable', 'sol-nu', 'surface_eau',
                'coniferes', 'feuillus', 'broussaille', 'vigne', 'pelouse', 'culture', 'terre_labouree']
        miou, ious = calc_miou(cm)
        tab = PrettyTable(['Class', 'mIou'])
        for i in range(self.num_classes):
            tab.add_row([classes[i], ious[i]])
        tab.add_row(["final mIoU", miou])
        with open(f"experiments/{self.config_name}/final_ious.txt", 'w') as f:
            f.write(str(tab))
        f.close()
        print(tab)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idx, images, targets = batch
        outputs = self.forward(images, idx)
        proba = torch.softmax(outputs["logits"], dim=1)
        out_batch = {}
        out_batch["preds"] =  torch.argmax(proba, dim=1)
        out_batch["img"] = images
        out_batch["id"] = idx
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