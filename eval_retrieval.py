import argparse, os, sys, pdb

import numpy as np
import scipy as sp
import scipy.stats
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from cirtorch.datasets.testdataset import configdataset
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.download import download_test
from cirtorch.utils.evaluate import compute_map
from cirtorch.utils.general import get_data_root

from cirtorchclone.imageretrievalnet import init_network
from utils import img_loader


datasets_names = ['roxford5k', 'rparis6k']
# datasets_names = ['roxford5k', 'rparis6k', 'holidays', 'copydays'] # holidays and copydays are not yet supported

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK', help="network off-the-shelf, in the format 'ARCHITECTURE-POOLING, eg: resnet18-gem or alexnet-gem or alexnet-mac or alexnet-rmac etc...")

# test options
parser.add_argument('--dataset', '-d', metavar='DATASETS', default='roxford5k', help="test dataset name: | ".join(datasets_names) + " (default: 'roxford5k')")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N', help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--image-resize', '-imresize', default=1024, type=int, metavar='N', help="maximum size of longer image side used for testing (default: 1024)")

# attack options
parser.add_argument('--dir-attack', metavar='DIR_ATTACK', default=None, help="directory where the attack images are saved")
parser.add_argument('--ext-attack', metavar='EXT_ATTACK', default=None, help="extension in which the attack images are saved")
parser.add_argument('--dir-cache', metavar='DIR_CACHE', default=None, help="directory where extracted vectors are cached")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N', help="gpu id used for testing (default: '0')")

# output text file
parser.add_argument('--log', default=None, help="text file saving results for the paper")

def main():
    print(">> Retrieval evaluation of attacks\n")

    args = parser.parse_args()

    # check if unknown dataset
    if args.dataset not in datasets_names:
        raise ValueError('Unsupported or unknown dataset: {}!'.format(args.dataset))

    # check if test dataset are downloaded and download if they are not
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # parse off-the-shelf parameters
    offtheshelf = args.network_offtheshelf.split('-')
    net_params = {'architecture':offtheshelf[0], 'pooling':offtheshelf[1],'local_whitening':False,'regional':False,'whitening':False,'pretrained':True}

    # load off-the-shelf network
    print(">> Loading off-the-shelf network: '{}'".format(args.network_offtheshelf))
    net = init_network(net_params)
    # print(">>>> loaded network: \n'{}'".format(net.meta_repr()))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(mean=net.meta['mean'],std=net.meta['std'])
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    # evaluate on test dataset
    dataset=args.dataset
    print('>> {}: Extracting...'.format(dataset))

    # prepare config structure for the test dataset
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    cfg.update({'qext_a':args.ext_attack})
    cfg.update({'dir_data_a':args.dir_attack})
    cfg.update({'dir_images_a':cfg['dir_data_a']})
    cfg.update({'qim_fname_a':config_qimname_a})

    # reduce number of queries for holidays and copydays
    if dataset.startswith('holidays') or dataset.startswith('copydays'):
        cfg['nq'] = 50
        cfg['gnd'] = cfg['gnd'][:cfg['nq']]

    images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
    qimages_a = [cfg['qim_fname_a'](cfg,i) for i in range(cfg['nq'])]

    try:
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
    except:
        bbxs = None  # for holidays and copydays

    # extract descriptors and cache or load cached ones
    print('>> {}: database images...'.format(dataset))
    network_fn = args.network_offtheshelf
    if args.dir_cache is not None:
        vecs_fn = os.path.join(args.dir_cache, '{}_{}_{}_{}_vecs.pth'.format(dataset, network_fn, args.image_size, args.image_resize))
    if os.path.isfile(vecs_fn):
        vecs = torch.load(vecs_fn)
        print('>> loaded cached descriptors from {}'.format(vecs_fn))
    else:
        vecs = extract_vectors_a(net, images, args.image_size, args.image_resize, transform)
        torch.save(vecs, vecs_fn)

    print('>> {}: standard query images...'.format(dataset))
    qvecs = extract_vectors_a(net, qimages, args.image_size, args.image_resize, transform, bbxs=bbxs)
    print('>> {}: attack query images...'.format(dataset))
    qvecs_a = extract_vectors_a(net, qimages_a, args.image_size, args.image_resize, transform)
    
    print('>> {}: evaluating for image resolution {}'.format(dataset, args.image_resize))
    # convert to numpy
    vecs = vecs.numpy()
    qvecs = qvecs.numpy()
    qvecs_a = qvecs_a.numpy()

    qip = (qvecs*qvecs_a).sum(axis=0)
    qip_mean, qip_std = qip.mean(), qip.std()
    print('>> {}: inner product (target,attack) mean: {:.3f}, std: {:.3f}'.format(dataset, qip_mean, qip_std))

    # search, rank, and print
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)
    maps, mprs, aps = compute_map_and_print(dataset, ranks, cfg['gnd'])

    # attack search, rank, and print
    scores = np.dot(vecs.T, qvecs_a)
    ranks = np.argsort(-scores, axis=0)
    maps_a, mprs_a, aps_a = compute_map_and_print(dataset + ' attack ({})'.format(args.ext_attack), ranks, cfg['gnd'])
    
    if dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        r1, r2, r3 = 100*maps[1], 100*maps_a[1], 100*(maps_a[1]-maps[1]) # medium protocol
    else:
        r1, r2, r3 = 100*maps[0], 100*maps_a[0], 100*(maps_a[0]-maps[0])

    print('\n*** Summary ***\n attack: {}\n test: {}-{}-{} \n  mean ip (target,attack): {:.3f}\n  mAP: org {:.2f} att {:.2f} dif {:.2f}\n'.format(args.dir_attack.split('/')[-2], dataset, args.network_offtheshelf, args.image_resize, qip_mean, r1, r2, r3))
    
    if args.log is not None:
        with open(args.log, 'a') as f:
            f.write('\n attack: {}\n test: {}-{}-{} \n  mean ip (target,attack): {:.3f}\n  mAP: org {:.2f} att {:.2f} dif {:.2f}\n\n'.format(args.dir_attack.split('/')[-2], dataset, args.network_offtheshelf, args.image_resize, qip_mean, r1, r2, r3))

          

def extract_vectors_a(net, images, image_size, image_resize, transform, bbxs=None, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        for i, input in enumerate(loader):
            input = input.cuda()
            input_t = nn.functional.interpolate(input, scale_factor=image_resize/image_size, mode='bilinear', align_corners=False)
            vecs[:, i] = net(input_t).cpu().data.squeeze()
            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10], doprint = True):
    
    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k') or dataset.startswith('instre') or dataset.startswith('holidays') or dataset.startswith('copydays'):
        map, aps, _, _ = compute_map(ranks, gnd)
        if doprint:
            print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))
        return [map], [-1], aps

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        if doprint:
            print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
            print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
        return [mapE, mapM, mapH], [mprE, mprM, mprH], apsM


# query filenames for attacks
def config_qimname_a(cfg, i):
    return os.path.join(cfg['dir_images_a'], cfg['qimlist'][i] + cfg['qext_a'])

if __name__ == '__main__':
    main()