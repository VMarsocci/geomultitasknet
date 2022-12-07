import torch
import torch.nn as nn
import numpy as np
from ..utils import *

class DoubleConv(nn.Module):
    """Double convolution with BatchNorm as an option
    -  conv --> (BatchNorm) --> ReLu
    -  conv --> (BatchNorm) --> ReLu
        [w,h,in_ch] -> [w,h,out_ch] -> [w,h,out_ch]

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(DoubleConv, self).__init__()
        if batch_norm is True:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),  # different from original U-Net (padding is set to 0)
                nn.BatchNorm2d(out_ch),                  # original U-Net does not contain batch normalisation
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),  # different from original U-Net (padding is set to 0)
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InputConv(nn.Module):
    """Input convolution, clone of DoubleConv
    """
    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(InputConv, self).__init__()
        self.double_conv = DoubleConv(in_ch, out_ch, batch_norm)

    def forward(self, x):
        x = self.double_conv(x)
        return x


class EncoderConv(nn.Module):
    """Encoder convolution stack

    - 2x2 max-pooling with stride 2 (for downsampling)
        [w,h,in_ch] ->> [w/2,h/2,in_ch]
    - double_conv

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(EncoderConv, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, batch_norm)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class DecoderConv(nn.Module):
    """Decoder convolution stack

    - deconvolution (*2) with stride 2 upscale
        [w,h,in_ch] -> [w*2,h*2,in_ch/2]
    - concatenation
        [w*2,h*2,in_ch/2] -> [w*2,h*2,in_ch/2+in_ch/2]
    - double_conv

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    bilinear : bool, optional
        enable bilinearity in DecoderConv, by default True
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(DecoderConv, self).__init__()
        # upconv divide number of channels by 2 and divide widh, height by 2 with stride=2

        # ConvTranspose2d reduces number of channels and width, height
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, batch_norm)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


class OutputConv(nn.Module):
    """Final layer:

    - convolution
        [w,h,in_ch] -> [w,h,out_ch]

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet = U-Net with reduced number of params (2M)

    Parameters
    ----------
    n_channels : int
        number of input channels
    n_classes : int
        number of output classes
    dropout : float
        if False no dropout
    """

    def __init__(self, n_channels, n_classes, drop_out = False):

        super(UNet, self).__init__()

        self.n_classes = n_classes
        self.name = "UNet"

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)

    def forward(self, x):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x1, x2, x5, x

########### GEO UNET
class GeoUNet(nn.Module):
    """UNet = U-Net with reduced number of params (2M)

    Parameters
    ----------
    n_channels : int
        number of input channels
    n_classes : int
        number of output classes
    dropout : float
        if False no dropout
    """

    def __init__(self, n_channels, n_classes, geoinfo = False, drop_out = False, 
                    use_time = False, use_geo = False, 
                    use_domains = False, use_coords_pos_enc = False, use_label_distr = False):

        super(GeoUNet, self).__init__()
        self.name = "GeoUNet"

        self.n_classes = n_classes
        self.geoinfo = geoinfo
        self.use_time = use_time
        self.use_geo = use_geo
        self.use_domains = use_domains
        self.use_coords_pos_enc = use_coords_pos_enc
        self.use_label_distr = use_label_distr

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)

        if self.use_time:
            self.hours_embedding = nn.Embedding(24, 256)
            self.months_embedding = nn.Embedding(12, 256)
            self.year_embedding = nn.Embedding(5, 256)

        if self.use_geo:
            if self.use_domains:              
                self.domain_embedding = nn.Embedding(13, 256) #50 for full domains
            else:
                if self.use_coords_pos_enc:
                    self.coords_embedding = nn.Sequential(
                      nn.Linear(in_features=256, out_features=256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=256, out_features=256),
                      )
                else:
                    self.coords_embedding = nn.Sequential(
                      nn.Linear(in_features=2, out_features=256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=256, out_features=256),
                      )
                
        if self.use_label_distr:
            self.label_distr_embedding = nn.Sequential(
                      nn.Linear(in_features=13, out_features=128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=128, out_features=256),
                      )
    
    def forward(self, x, idx):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)

        bs, c, h, w = x5.size()
        x5 = x5.view(bs, c, -1)

        coords, months, hours, years, domains, lab_distr = spatiotemporal_batches(idx, self.geoinfo, self.use_coords_pos_enc)

        if self.use_geo:
            if self.use_domains:
                emb_coords = self.domain_embedding(domains)
            else:
                emb_coords = self.coords_embedding(coords)
        else:
            emb_coords = torch.zeros((bs, h*w)).cuda()

        if self.use_time:
            emb_hours = self.hours_embedding(hours)
            emb_months = self.months_embedding(months)
            emb_years = self.year_embedding(years)
        else:
            emb_hours = torch.zeros((bs, h*w)).cuda()
            emb_months = torch.zeros((bs, h*w)).cuda()
            emb_years = torch.zeros((bs, h*w)).cuda()

        if self.use_label_distr:
            emb_label_distr = self.label_distr_embedding(lab_distr)
            # print(emb_label_distr.shape)
        else:
            emb_label_distr = torch.zeros((bs, h*w)).cuda()

        feat_maps = []

        for i in range(x5.size(-1)):
            x_ = x5[:,:,i].squeeze(1)
            # print(x_.shape)
            a = (x_ + emb_months + emb_hours + emb_years + emb_coords + emb_label_distr).unsqueeze(-1)
            feat_maps.append(a)
        x5 = torch.cat(feat_maps, dim = -1).reshape((bs,c,h,w))
        # print(emb_hours.shape, emb_months.shape, emb_years.shape, emb_coords.shape)
        # print(x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x1, x2, x5, x

class MultiTaskUNet(nn.Module):
    """UNet = U-Net with reduced number of params (2M)

    Parameters
    ----------
    n_channels : int
        number of input channels
    n_classes : int
        number of output classes
    dropout : float
        if False no dropout
    """

    def __init__(self, n_channels, n_classes, drop_out = False):

        super(MultiTaskUNet, self).__init__()

        self.n_classes = n_classes
        self.name = "MultiTaskUNet"

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        self.ch1 = EncoderConv(16, 32)
        self.ch2 = EncoderConv(32, 64)
        self.coords_predictor = nn.Sequential(
            nn.Linear(in_features=64*64*64, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            )

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)            

    def forward(self, x):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_2d = self.outc(x)

        x = self.ch1(x) #b, 32, 128, 128
        x = self.ch2(x) #b, 64, 64, 64
        x_coord = self.coords_predictor(x.view(x.size(0), -1))

        return x_2d, x_coord


class ConcatGeoUNet(nn.Module):
    """UNet = U-Net with reduced number of params (2M)

    Parameters
    ----------
    n_channels : int
        number of input channels
    n_classes : int
        number of output classes
    dropout : float
        if False no dropout
    """

    def __init__(self, n_channels, n_classes, geoinfo = False, drop_out = False, 
                    use_time = False, use_geo = False, 
                    use_domains = False, use_coords_pos_enc = False, use_label_distr = False):

        super(ConcatGeoUNet, self).__init__()
        self.name = "ConcatGeoUNet"

        self.n_classes = n_classes
        self.geoinfo = geoinfo
        # self.use_time = use_time
        self.use_geo = use_geo
        self.use_domains = use_domains
        self.use_coords_pos_enc = use_coords_pos_enc
        self.use_label_distr = use_label_distr

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # middle conv
        self.mid = OutputConv(512, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)

        if self.use_geo:
            if self.use_domains:              
                self.domain_embedding = nn.Embedding(13, 256) #50 for full domains
            else:
                if self.use_coords_pos_enc:
                    self.coords_embedding = nn.Sequential(
                      nn.Linear(in_features=256, out_features=256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=256, out_features=256),
                      )
                else:
                    self.coords_embedding = nn.Sequential(
                      nn.Linear(in_features=2, out_features=256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=256, out_features=256),
                      )
                
    def forward(self, x, idx):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)

        bs, c, h, w = x5.size()

        coords, months, hours, years, domains, lab_distr = spatiotemporal_batches(idx, self.geoinfo, self.use_coords_pos_enc)

        if self.use_domains:
            emb_coords = self.domain_embedding(domains)
        else:
            emb_coords = self.coords_embedding(coords)

        emb_coords = torch.reshape(emb_coords, (bs,c,1,1)).expand(bs,c,h,w)

        x5 = torch.cat([x5, emb_coords], dim = 1)

        x5 = self.mid(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x1, x2, x5, x

###########TIME AND GEO MULTI TASK
class GeoTimeTaskUNet(nn.Module):
    """UNet = U-Net with reduced number of params (2M)

    Parameters
    ----------
    n_channels : int
        number of input channels
    n_classes : int
        number of output classes
    dropout : float
        if False no dropout
    """

    def __init__(self, n_channels, n_classes, drop_out = False):

        super(GeoTimeTaskUNet, self).__init__()

        self.n_classes = n_classes
        self.name = "GeoTimeTaskUNet"

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        self.ch1 = EncoderConv(16, 32)
        self.ch2 = EncoderConv(32, 64)
        self.coords_predictor = nn.Sequential(
            nn.Linear(in_features=64*64*64, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            )

        self.time_predictor = nn.Sequential(
            nn.Linear(in_features=64*64*64, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=4),
            )

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)            

    def forward(self, x):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_2d = self.outc(x)

        x = self.ch1(x) #b, 32, 128, 128
        x = self.ch2(x) #b, 64, 64, 64
        x_coord = self.coords_predictor(x.view(x.size(0), -1))
        x_time = self.time_predictor(x.view(x.size(0), -1))

        return x_2d, x_coord, x_time

######KEEP IT SIMPLE FDM
class FDMUNet(nn.Module):

    def __init__(self, n_channels, n_classes, drop_out = False):

        super(FDMUNet, self).__init__()

        self.n_classes = n_classes
        self.name = "FDMUNet"

        # encoder
        self.inc = InputConv(n_channels, 16)
        self.down1 = EncoderConv(16, 32)
        self.down2 = EncoderConv(32, 64)
        self.down3 = EncoderConv(64, 128)
        self.down4 = EncoderConv(128, 256)
        # decoder
        self.up1 = DecoderConv(256, 128)
        self.up2 = DecoderConv(128, 64)
        self.up3 = DecoderConv(64, 32)
        self.up4 = DecoderConv(32, 16)

        # last layer
        self.outc = OutputConv(16, n_classes)

        # dropout
        self.drop_out = drop_out
        if self.drop_out:
            self.dropout = nn.Dropout(drop_out)            

    def forward(self, x, xt = torch.Tensor(1)):

        x1 = self.inc(x)
        if self.drop_out:
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if self.drop_out:
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if self.drop_out:
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if self.drop_out:
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if self.drop_out:
            x5 = self.dropout(x5)

        if xt.shape == x5.shape:
            x5 = torch_matching(x5, xt)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x1, x2, x5, x

################LOADING RIGHT MODEL
def choose_model(model_params, geo_data):
    if model_params["model_name"] == "unet":
        model =  UNet(n_channels = model_params['num_channels'], 
                n_classes = model_params['num_classes'], 
                drop_out = model_params['dropout'])
    elif model_params["model_name"] == "keepitsimple":
        model = FDMUNet(n_channels = model_params['num_channels'], 
           n_classes = model_params['num_classes'], 
           drop_out = model_params['dropout'])
    elif model_params["model_name"] == "mt_geo_time":
        model = GeoTimeTaskUNet(n_channels = model_params['num_channels'], 
           n_classes = model_params['num_classes'], 
           drop_out = model_params['dropout'])
    elif model_params["model_name"] == "concat_geounet":
        model = ConcatGeoUNet(n_channels = model_params['num_channels'], 
           n_classes = model_params['num_classes'], 
           geoinfo= geo_data,
           drop_out = model_params['dropout'],
           use_time= model_params['use_time'],
           use_geo= model_params['use_geo'],
           use_domains= model_params['use_domains'],
           use_coords_pos_enc = model_params['use_coords_pos_enc'],
           use_label_distr = model_params['use_label_distr'])
    elif model_params["model_name"] == "multitaskunet":
        model = MultiTaskUNet(n_channels = model_params['num_channels'], 
           n_classes = model_params['num_classes'], 
           drop_out = model_params['dropout'])
    elif model_params["model_name"] == "geounet":
        model = GeoUNet(n_channels = model_params['num_channels'], 
           n_classes = model_params['num_classes'], 
           geoinfo= geo_data,
           drop_out = model_params['dropout'],
           use_time= model_params['use_time'],
           use_geo= model_params['use_geo'],
           use_domains= model_params['use_domains'],
           use_coords_pos_enc = model_params['use_coords_pos_enc'],
           use_label_distr = model_params['use_label_distr']
           )  
    else:
        raise Exception("This model is not implemented")       

    return model
