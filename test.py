import torch
import torch.nn as nn
from pytorch_lightning import Trainer


import numpy as np
import shutil
from pathlib import Path
import yaml
import os
from argparse import ArgumentParser

from modules.augmentation import choose_training_augmentations, get_validation_augmentations
from modules.datamodule import DataModule
from modules.task_module import SegmentationTask
from modules.model import choose_model
from modules.optim import set_optimizer, set_scheduler
from modules.utils import get_geo_data, choose_loss
from modules.writer import PredictionWriter

def get_args():
    parser = ArgumentParser(description = "Hyperparameters", add_help = True)
    parser.add_argument('-c', '--config-name', type = str, help = 'YAML Config name', dest = 'CONFIG', default = 'baseline')
    parser.add_argument('-nw', '--num-workers', type = int, help = 'Number of workers', dest = 'NW', default = 12)
    parser.add_argument('-gpu', '--gpus_per_node', type = int, help = 'Number of GPUs per node', dest = 'GPUs', default = 1)
    parser.add_argument('-n', '--nodes', type = int, help = 'Number of nodes', dest = 'Ns', default = 1)
    parser.add_argument('-s', '--strategy', type = str, help = 'None if only one GPU, else ddp', dest = 'S', default = None)
    parser.add_argument('-p', '--predict', type = str, help = 'If True, do the predictions', dest = 'PREDICT', default = None)
    return parser.parse_args()

args = get_args()

manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
gpus_per_node = args.GPUs # set to 1 if mono-GPU
num_nodes = args.Ns # set to 1 if mono-GPU
strategy = args.S # Put this parameter to None if train on only one GPU or on CPUs. If multiple GPU, set to 'ddp'
num_workers = args.NW
config_name = args.CONFIG

# Load the configuration params of the experiment
print(f"Loading experiment {config_name}")
with open('config/'+config_name + ".yaml", "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

# Create path for the results
exp_directory = "experiments/" + config_name
os.makedirs(exp_directory, exist_ok=True)
out_file = exp_directory + '/' + exp_config['general']['test_id']
os.makedirs(out_file, exist_ok=True)
print(f"Logs and/or checkpoints will be stored on {exp_directory}")
shutil.copyfile('config/' + config_name + ".yaml", out_file + '/config.yaml')
print("Config file correctly saved!")

###########   AUGMENTATIONS    ##########
train_trans = choose_training_augmentations(exp_config)
val_trans = get_validation_augmentations(*exp_config['data']['val']['normalization'])   

###########  GET OTHER DATA ##############
geo_data = get_geo_data("../data/DATASET_DEF1_METADATA_train.json", "../data//DATASET_DEF1_METADATA_test.json")
metadata = exp_config["metadata"]

###########   DATAMODULE    ##########
dm = DataModule(
    path = exp_config['data']['path'],
    source_images_txt = exp_config['data']['train']['img_txt'], 
    source_masks_txt = exp_config['data']['train']['mask_txt'], 
    target_images_txt = exp_config['data']['val']['img_txt'], 
    target_masks_txt = exp_config['data']['val']['mask_txt'], 
    bands = exp_config['data']['bands'],
    train_augmentation= train_trans,
    valid_augmentation= val_trans,
    # cropsize = 256,
    geoinfo = False,
    batch_size = 1,
    num_workers = num_workers,
    drop_last = False,
    uda = exp_config['general']['uda']
    )

###########   MODEL    ##########
net = choose_model(exp_config['model'], geo_data)
ckpt_path = exp_config['model']['ckpt_path'] if exp_config['model']['ckpt_path'] else None

# #####TEMP LOADING
# pth = torch.load("experiments/net.pth", map_location='cuda:0')
# net.load_state_dict(pth)

###########   LOSS    ##########
criteria = choose_loss(exp_config['model'])

###########   OPTIMIZER AND SCHEDULER    ##########
optimizer = set_optimizer(exp_config['optim'], net)
scheduler = set_scheduler(exp_config['optim'], optimizer)

###########   SEGMENTATION    ##########
seg_module = SegmentationTask(
    model=net,
    num_classes=exp_config['model']['num_classes'],
    criteria=criteria,
    optimizer=optimizer,
    scheduler=scheduler,
    uda = exp_config['general']['uda'],
    geo_data = geo_data,
    metadata = metadata,
    config_name=config_name,
)

###########   CALLBACKS    ##########
writer_callback = PredictionWriter(        
    output_dir=os.path.join(out_file, "predictions"),
    write_interval="batch",
)

#### instanciation of prediction Trainer
trainer = Trainer(
    accelerator="gpu",
    devices=gpus_per_node,
    strategy=strategy,
    num_nodes=num_nodes,
    callbacks = [writer_callback],
    enable_progress_bar = True,
)

if __name__ == '__main__':
    print("+++++++++++++++++++++TESTING STAGE")
    metrics = trainer.test(seg_module, datamodule=dm, ckpt_path=ckpt_path)
    if args.PREDICT:
        print("+++++++++++++++++++++PREDICTING STAGE")
        trainer.predict(seg_module, datamodule=dm, ckpt_path=ckpt_path)