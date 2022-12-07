from .models.unet import UNet, FDMUNet, GeoTimeTaskUNet, ConcatGeoUNet, MultiTaskUNet, GeoUNet
from .models.deeplab import DeeplabV3p
from .models.resnetunet import UNetResNet

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
    elif model_params["model_name"] == "resunet18":
        model = UNetResNet(18, 
                n_classes = model_params['num_classes'], 
                n_channels = model_params['num_channels'], 
                num_filters=32, 
                dropout_2d=0.2,
                pretrained=True, 
                is_deconv=False)
    else:
        raise Exception("This model is not implemented")       

    return model
