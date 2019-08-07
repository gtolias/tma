from __future__ import print_function
import os,  sys, argparse, ast, time, pdb
import numpy as np
from PIL import Image
from scipy.misc import imsave

import torch

from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root

from cirtorchclone.imageretrievalnet import init_network
from tma import tma
from utils import img_loader, center_crop

# cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="roxford5k", help = 'roxford5k | rparis6k | holidays | copydays')
parser.add_argument('--carrier', type=str, default="flower", help = ' flower | sanjuan')
parser.add_argument('--mode', type=str, default="global", help = "global | tensor | hist")
parser.add_argument('--modellist', type=str, default="alexnet-mac")
parser.add_argument('--scales', type=str, default="[1024]", help = "[1024], [300,400,500,600,700,800,900,1024], [300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]")
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lam', type=float, default=.0)
parser.add_argument('--sigma-blur', type=float, default=0.3, help = "no blur if 0.0")
parser.add_argument('--gpu-id', type=str, default="0")
parser.add_argument('--variant', type=str, default="")

args = parser.parse_args()
args.scales = ast.literal_eval(args.scales)

# choose gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

# copy arguments
dataset, train_scales, iters, lr, lam, sigma_blur, mode, variant, carrier_fn = args.dataset, args.scales, args.iters, args.lr, args.lam, args.sigma_blur, args.mode, args.variant, args.carrier

# project folders
output_folder = "data/"
datasets_folder = get_data_root()+"/test/"

# attack-FCN / attack-pooling list
net_params = {'local_whitening':False,'regional':False,'whitening':False,'pretrained':True} # default cir network params
train_networks = []
for s in args.modellist.split("+"):
    net_params['architecture'] = s.split("-")[0] 
    train_networks.append(init_network(net_params))
    if mode == 'global': train_networks[-1].poolattack = s.split("-")[1:] 
for n in train_networks: n.eval(); n.cuda(); 

# output name
if mode == 'global': exp_name = dataset+"_"+("+".join([n.meta['architecture']+"-"+"-".join(n.poolattack) for n in train_networks]))
else: exp_name =  dataset+"_"+("+".join([n.meta['architecture']+"-"+mode for n in train_networks]))
if len(variant): variant= "+"+variant
exp_name+= "+"+str(train_scales).replace(" ","")+"+iter"+str(iters)+"+lr"+str(lr)+"+lam"+str(lam)+"+sigmablur"+str(sigma_blur)+"_"+carrier_fn+variant

# dataset config
cfg = configdataset(dataset, datasets_folder)
if dataset.startswith('holidays') or dataset.startswith('copydays'): cfg['nq'] = 50                                  # hard code holidays and copydays queries to first 50
if 'bbx' in cfg['gnd'][0].keys(): bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]                     # bounding boxes for roxford5k and rparis6k datasets
else: bbxs = None  
im_size = {'roxford5k':1024, 'rparis6k':1024, 'holidaysmanrot1024':1024, 'copydays':1024}                            # original image size
scale_factors = [x / im_size[dataset] for x in train_scales]                                                         # compute relative re-scaling factors

# log file
log = open(output_folder+"/log_"+exp_name+".txt", 'a')

# save folder for attacks
if not os.path.exists(output_folder+"/"+exp_name): 
    os.makedirs(output_folder+"/"+exp_name)

# params for rerun if no converge
max_trials, multiply_rate_iters, divide_rate_lr  = 10, 2, 5

total_time = 0
for i in range(cfg['nq']):
    query_fn = cfg['qimlist'][i]
    print(str(i)+" QUERY "+query_fn); log.write(str(i)+" QUERY "+query_fn+"\n"); log.flush()
    
    # load target (query) image
    if bbxs is not None:
        target_img = img_loader(cfg['qim_fname'](cfg,i), im_size[dataset], cfg['gnd'][i]['bbx']).type(torch.cuda.FloatTensor)
    else:
        target_img = img_loader(cfg['qim_fname'](cfg,i), im_size[dataset]).type(torch.cuda.FloatTensor)

    # attack
    t = time.time()
    trials = 0
    converged = False
    while not converged and trials < max_trials:
        carrier_img = img_loader("data/input/"+carrier_fn+".jpg", im_size[dataset]).type(torch.cuda.FloatTensor)
        carrier_img = center_crop(target_img, carrier_img)
        alr = lr / divide_rate_lr**trials # reducing lr after every failure
        aiters = int(iters * multiply_rate_iters**trials) # increase iterations after every failure
        attack_img, loss_perf, loss_distort, converged = tma(train_networks, scale_factors, target_img, carrier_img, mode = mode, num_steps = aiters, lr = alr, lam = lam, sigma_blur = sigma_blur, verbose = True) 
        trials += 1

    # time and log
    total_time += time.time()-t
    log.write("performance loss  {:6f} distortion loss {:6f} total loss {:6f}\n".format(loss_perf.item(), (loss_distort).item(), (loss_distort+loss_perf).item())); log.flush()
    if trials == max_trials: print("Failed...")

    # save the attack in png file    
    savefn = output_folder+exp_name+"/"+query_fn+'.png'
    if not os.path.exists(savefn[0:savefn.rfind('/')]): os.makedirs(savefn[0:savefn.rfind('/')])
    imsave(savefn,np.transpose(attack_img.cpu().numpy(), (2,3,1,0)).squeeze())    
    print("Attack saved in "+savefn+"\n"); sys.stdout.flush()

log.write("Average time per image {:6f}\n".format(total_time / cfg['nq'])); log.flush()
print("Average time per image {:6f}\n".format(total_time / cfg['nq']))    

log.close()
