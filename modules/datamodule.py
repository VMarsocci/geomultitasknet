from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import SupDataset, UnsupDataset, InfiniteDataLoader


class DataModule(LightningDataModule):

    def __init__(
        self, 
        path,
        source_images_txt, 
        source_masks_txt, 
        target_images_txt, 
        target_masks_txt, 
        bands = 'rgbirh',
        train_augmentation=None,
        valid_augmentation=None,
        cropsize = 256,
        geoinfo = False,
        batch_size = 25,
        num_workers = 0,
        drop_last = False,
        uda = False,
    ):
        super().__init__()
        self.path = path,
        self.source_images_txt = source_images_txt, 
        self.source_masks_txt = source_masks_txt, 
        self.target_images_txt = target_images_txt, 
        self.target_masks_txt = target_masks_txt, 
        self.bands = bands,
        self.train_augmentation= train_augmentation,
        self.valid_augmentation = valid_augmentation,
        self.cropsize = cropsize,
        self.geoinfo = geoinfo,
        self.batch_size = batch_size, 
        self.num_workers = num_workers, 
        self.drop_last = drop_last,
        self.uda = uda,

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.source_dataset = SupDataset(
                self.path[0],
                self.source_images_txt[0], 
                self.source_masks_txt[0], 
                bands = self.bands[0],
                augmentation= self.train_augmentation[0],
            )

            self.target_dataset = SupDataset(
                self.path[0],
                self.target_images_txt[0], 
                self.target_masks_txt[0], 
                bands = self.bands[0],
                augmentation= self.train_augmentation[0],
            )

            self.val_dataset = SupDataset(
                    self.path[0],
                    self.target_images_txt[0], 
                    self.target_masks_txt[0], 
                    bands = self.bands[0],
                    augmentation= self.valid_augmentation[0],
                )

        elif stage == "predict" or stage == "validate":
                self.val_dataset = SupDataset(
                    self.path[0],
                    self.target_images_txt[0], 
                    self.target_masks_txt[0], 
                    bands = self.bands[0],
                    augmentation= self.valid_augmentation[0],
                )

                self.test_dataset = SupDataset(
                    self.path[0],
                    self.target_images_txt[0], 
                    self.target_masks_txt[0], 
                    bands = self.bands[0],
                    augmentation= self.valid_augmentation[0],
                )

    def train_dataloader(self):
        if self.uda[0]:
            source_dataloader = DataLoader(
            dataset=self.source_dataset,
            batch_size=self.batch_size[0],
            shuffle=True,
            num_workers=self.num_workers[0],
            drop_last=True,
            )
            target_dataloader = InfiniteDataLoader(
            dataset=self.target_dataset,
            batch_size=self.batch_size[0],
            shuffle=True,
            num_workers=self.num_workers[0],
            drop_last=True,
            )
            loaders = {"source" : source_dataloader, 
                       "target" : target_dataloader}
            return loaders
        else:
            loaders = {"source" : DataLoader(
            dataset=self.source_dataset,
            batch_size=self.batch_size[0],
            shuffle=True,
            num_workers=self.num_workers[0],
            drop_last=self.drop_last[0],
            ) 
            }
            return loaders

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers[0],
            drop_last=self.drop_last[0],
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers[0],
            drop_last=self.drop_last[0],
        )