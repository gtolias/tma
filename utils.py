import random, pdb
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms.functional as TF

from cirtorch.datasets.datahelpers import imresize


def center_crop(query_img, carrier_img):
    q_i, q_c, q_w, q_h = query_img.size()
    c_i, c_c, c_w, c_h = carrier_img.size()
    left = (c_w - q_w)//2
    top = (c_h - q_h)//2
    return carrier_img[0,:,left:left+query_img.size(2),top:top+query_img.size(3)].unsqueeze(0)


# load image, optionally crop, and resize
def img_loader(image_name, imsize, bbx = None):
    img = Image.open(image_name)
    imsize_ = np.max((img.height,img.width))
    if bbx: img = img.crop(bbx)
    img = imresize(img, np.floor(1.0*np.max((img.height,img.width))*imsize/imsize_))
    return Variable(TF.to_tensor(img)).unsqueeze(0)


def reproduce(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_sim(networks, scales, target_img, carrier_img, attack_img):
    for network in networks:
        network.cuda()
        m = torch.FloatTensor(network.meta['mean']).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = torch.FloatTensor(network.meta['std']).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for scale in scales:
            print(network.meta['architecture']+"-"+network.meta['pooling']+" scale "+str(scale))    
            target = network(nn.functional.interpolate((target_img-m)/s, scale_factor=scale, mode='bilinear', align_corners=False)).squeeze()
            attack = network(nn.functional.interpolate((attack_img-m)/s, scale_factor=scale, mode='bilinear', align_corners=False)).squeeze()            
            carrier = network(nn.functional.interpolate((carrier_img-m)/s, scale_factor=scale, mode='bilinear', align_corners=False)).squeeze()

            print("T2A: {:.3f} C2A: {:.3f} C2T: {:.3f}".format(target.dot(attack).item(), carrier.dot(attack).item(), carrier.dot(target).item()))
        network.cpu()