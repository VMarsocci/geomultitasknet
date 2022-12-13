from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import torch
import os
import tifffile as tiff
import numpy as np

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
            cropsize = 256,
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
            if self.stage == "train":
                h = np.random.randint(0, self.cropsize)
                w = np.random.randint(0, self.cropsize)
                image = image[h:h+self.cropsize, w:w+self.cropsize, :]
                mask = mask[h:h+self.cropsize, w:w+self.cropsize]
            else:
                h_cps = int(self.cropsize/2)
                hc = wc = 256
                image = image[hc-h_cps:hc+h_cps, wc-h_cps:wc+h_cps, :]
                mask = mask[hc-h_cps:hc+h_cps, wc-h_cps:wc+h_cps]
        
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
        
# class UnsupDataset(BaseDataset):
    
#     def __init__(
#             self, 
#             path,
#             images_txt, 
#             bands = 'rgbirh',
#             augmentation=None,
#             cropsize = 256,
#             crop_indexes = False #for noisy student training
#             ):

#         with open(images_txt) as f:
#             lines = f.readlines()
            
#         self.images_fps = sorted([line.strip() for line in lines]) 
#         self.augmentation = augmentation
#         self.bands = bands
#         self.data_path = path
#         self.cropsize = cropsize
#         self.crop_indexes = crop_indexes
    
#     def __getitem__(self, i):
        
#         # read data
#         image = tiff.imread(os.path.join(self.data_path, self.images_fps[i]))
        
#         image = load_bands(image, self.bands)
        
#         #random crop of the image
#         if self.cropsize:
#             self.h = np.random.randint(0, self.cropsize)
#             self.w = np.random.randint(0, self.cropsize)
#             image = image[self.h:self.h+self.cropsize, self.w:self.w+self.cropsize, :]

#         # apply augmentations
#         if self.augmentation:
#             image = self.augmentation(image = image)['image']

#         if self.crop_indexes:            
#             return self.images_fps[i], image, self.h, self.w
#         else:
#             return self.images_fps[i], image
        
#     def __len__(self):
#         return len(self.images_fps)
    
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