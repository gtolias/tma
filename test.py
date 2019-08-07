from __future__ import print_function
import sys, time, copy, os, pdb, argparse, ast
from scipy.misc import imsave
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.utils.model_zoo import load_url

from cirtorch.layers import pooling
from cirtorch.layers import normalization
from cirtorch.datasets.datahelpers import imresize

from cirtorchclone.imageretrievalnet import init_network

from tma import tma
from utils import *

gpu_id = '0'

train_scales = [300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]
test_scales = [1024]
iters = 100 
lr = 0.01 
lam = 0	
sigma_blur = 0.3

carrier_fn = 'data/input/flower.jpg'
# carrier_fn = 'data/input/sanjuan.jpg'
target_fn = 'data/input/notredame.jpg'

mode = 'hist'
pool = mode 
# mode = 'global'
# pool = 'gem'
arch = 'alexnet'
modellist = arch+"-"+pool

testarch = 'alexnet'
testpool = 'gem'
testmodellist = testarch+"-"+testpool

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

output_folder = 'data/'

# list of training networks / pooling
net_params = {'local_whitening':False,'regional':False,'whitening':False,'pretrained':True} # default cir network params

test_networks = []
for s in testmodellist.split("+"):
    net_params['architecture'] = s.split("-")[0] 
    for s2 in testmodellist.split("-")[1:]:
        net_params['pooling'] = s2
        test_networks.append(init_network(net_params))

train_networks = []
for s in modellist.split("+"):
    net_params['architecture'] = s.split("-")[0]
    train_networks.append(init_network(net_params))
    if mode == 'global': train_networks[-1].poolattack = s.split("-")[1:] 

for n in train_networks: n.eval(); n.cuda(); 

imsize = 1024
train_scale_factors = [x / imsize for x in train_scales]
test_scale_factors = [x / imsize for x in test_scales]

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
for n in train_networks: n.cuda(); n.eval() 
for n in test_networks: n.eval()

def loader(image_name, im_size):
    return Variable(TF.to_tensor(imresize(Image.open(image_name), im_size))).unsqueeze(0)

target_img = img_loader(target_fn, imsize).type(dtype)
carrier_img = img_loader(carrier_fn, imsize).type(dtype)

carrier_img = center_crop(target_img, carrier_img)
carrier_img_org = carrier_img.clone().clamp_(0, 1)

print("TARGET "+target_fn+" CARRIER " + carrier_fn)

t = time.time()        
attack_img = tma(train_networks, train_scale_factors, target_img, carrier_img, mode = mode, num_steps = iters, lr = lr, lam = lam, sigma_blur = sigma_blur, verbose = True)[0]
print("Elapsed time {:.4f}\n".format(time.time()-t))

# save to disk
img2save = np.transpose(attack_img.cpu().numpy(), (2,3,1,0)).squeeze()
imsave(output_folder+'/attack_image.png',img2save)
img_df = (attack_img-carrier_img_org)
img_df = (img_df-img_df.min()) / (img_df.max()-img_df.min())
img2save = np.transpose(img_df.cpu().numpy(), (2,3,1,0)).squeeze()
imsave(output_folder+'/attack_carrier_diff.png',img2save)

print("Evaluate descriptor similarity")
eval_sim(test_networks, test_scale_factors, target_img, carrier_img_org, attack_img)
