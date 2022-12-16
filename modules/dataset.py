from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import torch
import os
import tifffile as tiff
import numpy as np
import random
import cv2

class SupDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
    """
    
    def __init__(
            self, 
            path,
            images_txt, 
            masks_txt, 
            bands = 'rgbirh',
            augmentation=None,
            cropsize = False,
            geoinfo = False,
            stage = "train"
            ):

        with open(images_txt) as f:
            lines = f.readlines()
            
        self.images_fps = sorted([line.strip() for line in lines])  
        
        with open(masks_txt) as f:
            lines = f.readlines()
        self.masks_fps = sorted([line.strip() for line in lines])   
        
        self.data_path = path
        self.bands = bands
        self.augmentation = augmentation
        self.cropsize = cropsize
        self.geoinfo = geoinfo
        self.stage = stage

    def __getitem__(self, i):
        
        # read data
        image = tiff.imread(os.path.join(self.data_path,self.images_fps[i]))
        mask = tiff.imread(os.path.join(self.data_path,self.masks_fps[i]))
        
        # print(mask.dtype, mask.shape)
        
        image = load_bands(image, self.bands) #select only bands of interest
        
        mask[mask==19]=0
        mask[mask==18]=0 
        mask[mask==17]=0 
        mask[mask==16]=0 
        mask[mask==15]=0
        mask[mask==14]=0
        mask[mask==13]=0

        #random crop of the image and the mask
        if self.cropsize:
            # if self.stage == "train":
            #     # h = np.random.randint(0, self.cropsize)
            #     # w = np.random.randint(0, self.cropsize)
            #     # image = image[h:h+self.cropsize, w:w+self.cropsize, :]
            #     # mask = mask[h:h+self.cropsize, w:w+self.cropsize]
            image, mask = random_crop(image, mask, self.cropsize)
            # else:
                # if self.four_crops:

                # image = four_crops(image)

                # else:
                #     h_cps = int(self.cropsize/2)
                #     hc = wc = 256
                #     image = image[hc-h_cps:hc+h_cps, wc-h_cps:wc+h_cps, :]
                #     mask = mask[hc-h_cps:hc+h_cps, wc-h_cps:wc+h_cps]
                # image = cv2.resize(image, (self.cropsize, self.cropsize), interpolation = cv2.INTER_NEAREST)
                # mask = cv2.resize(mask, (self.cropsize, self.cropsize), interpolation = cv2.INTER_NEAREST)

        
        # apply augmentations
        if self.augmentation:
            pair = self.augmentation(image = image, mask = mask)
            image, mask = pair['image'], pair['mask']

        if self.geoinfo:
            coords = torch.unsqueeze(pos_enc(self.images_fps[i], self.geoinfo), -1)
            coords = torch.unsqueeze(coords.expand(256, 256), 0).float()
            image = torch.cat((image,coords), dim = 0)
        
        return self.images_fps[i], image, mask
        
    def __len__(self):
        return len(self.images_fps)
        
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
        
def load_bands(img, tag = 'rgbirh'):
    if tag == 'rgbirh':
        return img
    elif tag == 'rgb':
        return img[:,:,:3]
    elif tag == 'rgbir':
        return img[:,:,:4]
    else:
        return img


def pos_enc(img_name, info_diz):

    key = img_name.split("/")[1] + "-" + img_name.split("/")[2] + "-" + img_name.split("/")[-1].split(".")[0]

    x = info_diz[key]["patch_centroid_x"] - 489353.59 #center coordinate for EPSG:2154
    y = info_diz[key]["patch_centroid_y"] - 6587552.2 #center coordinate for EPSG:2154

    d= int(256/2)
    d_i=np.arange(0,d/2)
    freq=1/(10000**(2*d_i/d))
    enc=np.zeros(d*2)
    enc[0:d:2]=np.sin(x * freq)
    enc[1:d:2]=np.cos(x * freq)
    enc[d::2]=np.sin(y * freq)
    enc[d+1::2]=np.cos(y * freq)

    return torch.tensor(enc)

def crop_or_resize(image, mask, cropsize):
  n = random.randint(1,2)
  if n==1:
    choice = random_crop(image, mask, cropsize)   
  if n ==2:
    choice = im_resize(image, mask, cropsize)
  return choice

def im_resize(image, mask, cropsize):
    image = cv2.resize(image, (cropsize, cropsize), interpolation = cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (cropsize, cropsize), interpolation = cv2.INTER_NEAREST)
    return image, mask

def random_crop(image, mask, cropsize):
    h = np.random.randint(0, cropsize)
    w = np.random.randint(0, cropsize)
    image = image[h:h+cropsize, w:w+cropsize, :]
    mask = mask[h:h+cropsize, w:w+cropsize]
    return image, mask

# def four_crops(image, cropsize = 265):
#     im0 = image[0:256, 0:256, :]
#     im1 = image[256:512, 0:256, :]
#     im2 = image[0:256, 256:512, :]
#     im3 = image[256:512, 256:512, :]
#     # m0 = mask[0:256, 0:256]
#     # m1 = mask[256:512, 0:256]
#     # m2 = mask[0:256, 256:512]
#     # m3 = mask[256:512, 256:512]
#     return [im0, im1, im2, im3]