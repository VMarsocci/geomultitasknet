import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from PIL import ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import pathlib
import json
import random

# from .metrics import *

#### CORAL LOSS
class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.
    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by
    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))
    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by
    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        l = len(f_s)
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        del f_s
        del f_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (l - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (l - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff
        
#### DANN FUNCTIONS
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=256 * 16 * 16, out_features=100), #with img 512*512 put 512*32*32, with 256 512*16*16
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x, dim = 1)
        
        
def plot_grad_flow(named_parameters, epoch, graph_directory):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.figure(figsize=(25,10))
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=2, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=2, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=3, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=6),
                Line2D([0], [0], color="b", lw=6),
                Line2D([0], [0], color="k", lw=6)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(pathlib.Path(graph_directory) / str(epoch))
    plt.close()


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input, target):
        G = gram_matrix(input)
        target = gram_matrix(target)
        loss = F.mse_loss(G, target)
        return loss

def create_spatiotempoinfo(img_names, info_diz):

  batch = []
  for img_name in img_names:

    key = img_name.split("/")[1] + "-" + img_name.split("/")[2] + "-" + img_name.split("/")[-1].split(".")[0]

    x = info_diz[key]["patch_centroid_x"] - 489353.59 #center coordinate for EPSG:2154
    y = info_diz[key]["patch_centroid_y"] - 6587552.2 #center coordinate for EPSG:2154
    year = int(info_diz[key]["date"].split('-')[0]) - 2018 #norm with the minimum
    month = int(info_diz[key]["date"].split('-')[1])
    hour = int(info_diz[key]["time"].split('h')[0])
    if int(info_diz[key]["time"].split('h')[1]) > 30: #approx the hour
      hour += 1

    batch.append(torch.tensor([x, y, year, month, hour]))

  return torch.stack(batch)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

#KEEP IT SIMPLE
def torch_matching(source, reference):
    """ Run all transformation steps """
    # 1.) reshape to feature matrix (H*W,C)
    feature_mat_src = torch_get_feature_matrix(source)
    feature_mat_ref = torch_get_feature_matrix(reference)

    # 2.) center (subtract mean)
    feature_mat_src, _ = torch_center_image(feature_mat_src)
    feature_mat_ref, reference_mean = torch_center_image(feature_mat_ref)

    # 3.) whitening: cov(feature_mat_src) = I
    feature_mat_src_white = torch_whitening(feature_mat_src)

    # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
    feature_mat_src_transformed = torch_covariance_transformation(feature_mat_src_white, feature_mat_ref)

    # 5.) Add reference mean
    feature_mat_src_transformed += reference_mean

    # 6.) Reshape
    result = feature_mat_src_transformed.reshape(source.shape)

    return result

def torch_get_feature_matrix(image):
    """ Reshapes an image (H, W, C) to
    a feature vector (H * W, C)
    :param image: H x W x C image
    :return feature_matrix: N x C matrix with N samples and C features
    """
    feature_matrix = torch.reshape(image, (-1, image.shape[-1]))
    return feature_matrix

def torch_center_image(image):
    """ Centers the image by removing mean
    :returns centered image and original mean
    """
    image = torch.clone(image)
    image_mean = torch.mean(image, dim=0)
    image -= image_mean
    return image, image_mean

def torch_whitening(feature_mat):
    """
    Transform the feature matrix so that cov(feature_map) = Identity or
    if the feature matrix is one dimensional so that var(feature_map) = 1.
    :param feature_mat: N x C matrix with N samples and C features
    :return feature_mat_white: A corresponding feature vector with an
    identity covariance matrix or variance of 1.
    """
    if feature_mat.shape[1] == 1:
        variance = torch.var(feature_mat)
        feature_mat_white = feature_mat / torch.sqrt(variance)
    else:
        data_cov = cov(feature_mat, rowvar = False)
        u_mat, s_vec, _ = torch.linalg.svd(data_cov)
        sqrt_s = torch.diag(torch.sqrt(s_vec))
        feature_mat_white = (feature_mat @ u_mat) @ torch.linalg.inv(sqrt_s)
    return feature_mat_white

def torch_covariance_transformation(feature_mat_white, feature_mat_ref):
    """
    Transform the white (cov=Identity) feature matrix so that
    cov(feature_mat_transformed) = cov(feature_mat_ref). In the 2d case
    this becomes:
    var(feature_mat_transformed) = var(feature_mat_ref)
    :param feature_mat_white: input with identity covariance matrix
    :param feature_mat_ref: reference feature matrix
    :return: feature_mat_transformed with cov == cov(feature_mat_ref)
    """
    if feature_mat_white.shape[1] == 1:
        variance_ref = torch.var(feature_mat_ref)
        feature_mat_transformed = feature_mat_white * torch.sqrt(variance_ref)
    else:
        covariance_ref = cov(feature_mat_ref, rowvar = False)
        u_mat, s_vec, _ = torch.linalg.svd(covariance_ref)
        sqrt_s = torch.diag(torch.sqrt(s_vec))
        feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
    return feature_mat_transformed

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def spatiotemporal_batches(img_names, 
                            info_diz, 
                            pos_enc_coords = False, 
                            circle_encoding = False,
                            encoding_freq = 10000,
                            geo_noise = False):

  coord_batch = []
  month_batch = []
  hour_batch = []
  year_batch = []
  domain_batch = []
  lab_distr_batch = []

#   cat = {'D004_2021': 0, 'D058_2020': 1, 'D070_2020': 2, 'D078_2021': 3, 'D016_2020': 4, 'D013_2020': 5, 'D072_2019': 6, 'D067_2021': 7,'D021_2020': 8, 'D080_2021': 9, 
#         'D033_2021': 10, 'D074_2020': 11, 'D091_2021': 12, 'D017_2018': 13, 'D044_2020': 14, 'D006_2020': 15,'D066_2021': 16, 'D046_2019': 17,'D051_2019': 18, 'D052_2019': 19,
#         'D060_2021': 20, 'D077_2021': 21,'D041_2021': 22, 'D030_2021': 23,'D031_2019': 24, 'D014_2020': 25, 'D081_2020': 26, 'D023_2020': 27, 'D009_2019': 28, 'D008_2019': 29, 
#         'D007_2020': 30, 'D032_2019': 31, 'D086_2020': 32, 'D035_2020': 33,'D029_2021': 34, 'D034_2021': 35, 'D063_2019': 36, 'D055_2018': 37, 'D038_2021': 38, 'D049_2020': 39, 
#         'D075_2021': 40, 'D026_2020': 41, 'D064_2021': 42, 'D022_2021': 43, 'D071_2020': 44, 'D085_2019': 45,'D012_2019': 46,'D076_2019': 47,'D083_2020': 48,'D068_2021': 49}

  cat = {"D064_2021" : 0, "D068_2021" : 1, "D071_2020" : 2, "D006_2020" : 3, "D008_2019" : 4, "D013_2020" : 5,"D017_2018" : 6, 
        "D023_2020" : 7, "D029_2021" : 8, "D033_2021" : 9, "D058_2020": 10,"D067_2021": 11, "D074_2020" : 12}

# distr label domains
  lab_distr_d = {"D064_2021" : [0.0003647796201034331, 0.06452758359237457, 0.06800279214348591, 0.14679691368425396, 0.18834572805485256, 
                        0.09078354311660981, 0.01652554364271567, 0.12179119271291813, 0.08433024661641725, 0.0061088110695422535, 
                        0.18021248401050835, 0.01684239562128631, 0.015367986114931778],
         "D068_2021" : [0.004719607930787852, 0.06526496135013204, 0.07535856864821742, 0.12260942485970511, 0.007733042542363557, 
                        0.057821044921875, 0.006844044000330105, 0.19354975310849473, 0.035361164791483275, 0.06611181124834947, 
                        0.2357878263231734, 0.09719313070807659, 0.031645619567011445], 
         "D071_2020" : [0.0017459106445312498, 0.05118809170193142, 0.05566404554578993, 0.10303836398654515, 0.0003597598605685764, 
                        0.02859917534722222, 0.00203826904296875, 0.1325440385606554, 0.032226003011067705, 0.09877537197536893, 
                        0.35038882785373265, 0.1280748070610894, 0.015357335408528646], 
         "D006_2020" : [0.03492800215600242, 0.12720996050767497, 0.0802984404228103, 0.19650017751774318, 0.13191305885852223, 
                        0.0366021728515625, 0.059666593309859156, 0.10551761466012874, 0.09073230797136334, 0.003184088048800616, 
                        0.10934297803422095, 0.018314939686949822, 0.0057896659743617955],
         "D008_2019" : [0.007713683222381163, 0.09044400067396567, 0.07191997689260564, 0.13460382166043133, 0.0002639125770246479, 
                        0.0708207509215449, 0.02707674268265845, 0.18405594086982835, 0.028644924969740318, 0, 
                        0.17013242909606074, 0.2008037470428037, 0.013520069390955106], 
         "D013_2020" : [0.00011058862658514493, 0.05202315620754076, 0.16280777640964675, 0.14575036974920744, 0.0928029732082201, 
                        0.09952922434046649, 0.0491379005321558, 0.058422267747961956, 0.19792177062103714, 0.02897713923799819, 
                        0.08007818691972374, 0.013082390384397645, 0.019356256015058877],
         "D017_2018" : [0.000976715087890625, 0.10267012702094185, 0.0804528554280599, 0.15790524800618488, 0.03490570068359375, 
                        0.12083129035101997, 0.013971896701388889, 0.10339057922363282, 0.081606691148546, 0.043872511121961805, 
                        0.19314151340060765, 0.06627487182617188, 0], 
         "D023_2020" : [0.0003672281901041667, 0.01062700907389323, 0.037166239420572914, 0.038042399088541665, 4.887898763020833e-05, 
                        0.060361811319986976, 0.08545280456542968, 0.17749099731445311, 0.10059984842936198, 0, 
                        0.16132921854654947, 0.32851356506347656, 0],
         "D029_2021" : [0.004756734636094835, 0.08399313184950087, 0.04870170593261719, 0.17258464813232421, 0.028876094818115233, 
                        0.10229706658257379, 0.026563387976752388, 0.10788974126180013, 0.06514180077446832, 0.0004709879557291667, 
                        0.20942507637871635, 0.09209336174858941, 0.0572062619527181],
         "D033_2021" : [0.0012820993342869719, 0.08024832712092869, 0.1257984687912632, 0.1593204637984155, 0.057976908079335386, 
                        0.05922734650088028, 0.02655764297700264, 0.08596138107944543, 0.07612547968474911, 0.1250477793518926, 
                        0.1290090899400308, 0.037058784592319545, 0.03638622874944982],
         "D058_2020": [0.0023912556966145834, 0.03721769544813368, 0.061170247395833335, 0.0755106692843967, 0.014330079820421006, 
                        0.04607422722710503, 0.008577635023328993, 0.16849986606174044, 0.04217371622721354, 0.03822308858235677, 
                        0.15391856723361544, 0.3192870076497396, 0.03262594434950087],
         "D067_2021": [0.002779558706974638, 0.14089519113734147, 0.0812784034618433, 0.2273738076030344, 3.6541482676630437e-05, 
                        0.0510047802026721, 0.011937547766644022, 0.1469181403560915, 0.012665069137794385, 0.07098500127377717, 
                        0.15250635119451994, 0.03385585619055707, 0.06776375148607337],
         "D074_2020" : [0.006011663698682598, 0.06745630600873162, 0.047696437461703434, 0.11564254461550245, 0.07988143621706495, 
                        0.08252154181985294, 0.07511855181525735, 0.17171317306219364, 0.018746589211856617, 0.001415118049172794, 
                        0.23189109652650122, 0.09536426020603554, 0.006541281307444853]}

  for img_name in img_names:

    key = img_name.split("/")[1] + "-" + img_name.split("/")[2] + "-" + img_name.split("/")[-1].split(".")[0]

    x = info_diz[key]["patch_centroid_x"] - 489353.59 #center coordinate for EPSG:2154
    y = info_diz[key]["patch_centroid_y"] - 6587552.2 #center coordinate for EPSG:2154
    if geo_noise:
        if random.randint(0,1) == 0:
            x += random.randint(0, geo_noise)
            y += random.randint(0, geo_noise)
        elif random.randint(0,1) == 1:
            x -= random.randint(0, geo_noise)
            y -= random.randint(0, geo_noise)
    year = int(info_diz[key]["date"].split('-')[0]) - 2018 #norm with the minimum
    domain = int(cat[info_diz[key]["domain"]])
    month = int(info_diz[key]["date"].split('-')[1])-1
    hour = int(info_diz[key]["time"].split('h')[0])
    if int(info_diz[key]["time"].split('h')[1]) > 30: #approx the hour
      hour += 1
    lab_distr = lab_distr_d[info_diz[key]["domain"]]

    if pos_enc_coords:
        d= int(256/2)
        d_i=np.arange(0,d/2)
        freq=1/(encoding_freq**(2*d_i/d))
        enc=np.zeros(d*2)
        enc[0:d:2]=np.sin(x * freq)
        enc[1:d:2]=np.cos(x * freq)
        enc[d::2]=np.sin(y * freq)
        enc[d+1::2]=np.cos(y * freq)
        coord_batch.append(torch.tensor(enc).float())
    else:
        coord_batch.append(torch.tensor([x, y]))

    if circle_encoding:
        hour_ = hour*15/57.2958
        hour = [np.sin(hour_), np.cos(hour_)]

        month_ = month*30/57.2958
        month = [np.sin(month_), np.cos(month_)]
        hour_batch.append(torch.tensor(hour).float())
        month_batch.append(torch.tensor(month).float()) 
    else:
        hour_batch.append(torch.tensor(hour))
        month_batch.append(torch.tensor(month))  

    lab_distr_batch.append(torch.tensor(lab_distr))
    year_batch.append(torch.tensor(year))
    domain_batch.append(torch.tensor(domain))

  return torch.stack(coord_batch).cuda(), torch.stack(month_batch).cuda(), torch.stack(hour_batch).cuda(), torch.stack(year_batch).cuda(), torch.stack(domain_batch).cuda(), torch.stack(lab_distr_batch).cuda()

def get_geo_data(path_train, path_test):
    f1 = open(path_train)
    train_data = json.load(f1)
    f2 = open(path_test) 
    test_data = json.load(f2)
    geo_data = train_data | test_data
    return geo_data

def choose_loss(params):
    criteria = {}
    criteria["segmentation"] = nn.CrossEntropyLoss(ignore_index=params['class_ignored'])
    if params["constraint_name"] in ("gram", "style"):
        criteria["constraint"] = StyleLoss()
        criteria["constraint_name"] = "style_loss"
    elif params["constraint_name"] == "coral":
        criteria["constraint"] = CorrelationAlignmentLoss()
        criteria["constraint_name"] = "coral"
    elif params["constraint_name"] == "cosine_similarity":
        criteria["constraint"] = nn.CosineSimilarity(dim=1, eps=1e-6)
        criteria["constraint_name"] = "cosine_similarity"
    elif params["constraint_name"] == "multitask_strategy":
        criteria["constraint"] = nn.MSELoss()
        criteria["constraint_name"] = "multitask_strategy"
        criteria["mt_time"] = params["mt_time"]
    else:
        criteria["constraint"] = False
        criteria["constraint_name"] = False
        print("Constraint loss set to False")

    criteria["constraint_weight"] = params["weight"]
    return criteria

def calc_miou(cm_array):
    m = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = (np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))*100).round(2)
    m = (np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()).round(2)

    return m.astype(float), ious 