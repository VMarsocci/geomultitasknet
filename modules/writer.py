import os
from pickletools import uint8
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
# try: 
#     from pytorch_lightning.utilities.distributed import rank_zero_only 
# except ImportError:
#     from pytorch_lightning.utilities.rank_zero import rank_zero_only 
from PIL import Image
import torch



def segment_map_to_rgb_color_image(color_group, segment_map):
        
        rgb_color_image = np.zeros((len(segment_map[0]), len(segment_map[1]), 3), dtype=np.uint8)
    
        for i in range(np.size(segment_map, axis=0)):
             for j in range(np.size(segment_map, axis=1)):
                 idx = segment_map[i][j]
                 for p in range(len(color_group[0])):  
                     rgb_color_image[i][j][p] = color_group[idx][p]                    
                    
        return rgb_color_image

color_group = [
    [0, 0, 0],      #autres
    [241, 91, 181], #batiment
    [229,229,229],  #zone-permeable
    [157, 2, 8],    #zone-impermeable
    [55, 6, 23],    #sol-nu
    [67, 97, 238],  #surface_eau
    [19, 42, 19],   #coniferes
    [82, 183, 136], #feuillus
    [232, 93, 4],   #broussaille
    [114, 9, 183],  #vigne
    [79, 119, 45],  #pelouse
    [244, 140, 6],  #culture
    [250, 163, 7],  #terre_labouree
]

class PredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        output_dir,
        write_interval,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):

        preds, filename =prediction["preds"], prediction["id"][0]
        preds = torch.squeeze(preds)
        preds = segment_map_to_rgb_color_image(color_group, preds.cpu().numpy())
        filename = filename.replace('IMG', 'PRED').replace("tif", "png").replace("/", "_")
        output_file = str(self.output_dir+'/'+ filename)
        Image.fromarray(preds).save(output_file)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )
