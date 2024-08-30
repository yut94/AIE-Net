import argparse

import os
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.AIENet import CVTEnhancer
from dataset import DataSetTest
# from loss import *
import utils
import cv2
from PIL import Image


parser = argparse.ArgumentParser("CVTEnhancer")
parser.add_argument('--gpu_id', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--image_dir', type=str, default='./data/NPE')
parser.add_argument('--exp_dir', type=str, default='./exp/cvt_enhancer9', help='location of the data corpus')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

weight_path = os.path.join(args.exp_dir, 'model_epochs/best_psnr_weights.pt')

result_path = args.exp_dir + '/result_t_dual/NPE/'
os.makedirs(result_path, exist_ok=True)

logger = utils.create_logger(result_path)

np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def btf_sigmoid(x, illu, a=0.6, b=0.9):
    k = 1. / (illu+1e-6)
    # k = torch.minimum(k, torch.ones_like(k)*7)
    return torch.pow(k, b)*x*(1+a)/(torch.pow(k, b)*x-x+1+a)

def preferred(x, illu, a=4.35, b=0.14):
    k = 1. / (illu+1e-6)
    cf = torch.pow(x, 1/b)
    ka = torch.pow(k, a)
    return torch.pow(cf*ka/(cf*(ka-1)+1), b)

def BetaGamma(x, illu, a=-0.3293, b=1.1258):
    k = 1. / (illu+1e-6)
    beta = torch.exp(1.-torch.pow(k, a))*b
    gamma = torch.pow(k, a)
    return torch.pow(x, gamma)*beta

def Gamma(x, illu, a=0.8):
    k = 1. / (illu+1e-6)
    return x*torch.pow(k, 0.8)

def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    """perform the exposure fusion method used in the DUAL paper.

    Arguments:
        im {np.ndarray} -- input image to be enhanced.
        under_ex {np.ndarray} -- under-exposure corrected image. same dimension as `im`.
        over_ex {np.ndarray} -- over-exposure corrected image. same dimension as `im`.

    Keyword Arguments:
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})

    Returns:
        np.ndarray -- the fused image. same dimension as `im`.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images

def test():
    test_dataset = DataSetTest(img_dir=args.image_dir)
    logger.info("The number of valid images = %d", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               pin_memory=True, num_workers=0, shuffle=False)

    model = CVTEnhancer().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        for _, (img, image_name) in enumerate(test_loader):
            img = img.to(device)
            h,w = img.size(2), img.size(3)
            num_down=1
            win_size = 8
            padh = int(math.ceil(h/2**num_down/win_size)*win_size*2**num_down-h) if h/2**num_down%win_size != 0 else 0
            padw = int(math.ceil(w/2**num_down/win_size)*win_size*2**num_down-w) if w/2**num_down%win_size != 0 else 0
            pad_img = F.pad(img, pad=(padw//2, padw-padw//2, padh//2, padh-padh//2))
            print(image_name, ":", img.shape, "===>>", pad_img.shape)

            over_illu, over_ref = model(pad_img)
            over_illu = over_illu[:,:,padh//2:padh//2+h, padw//2:padw//2+w]
            over_ref = over_ref[:,:,padh//2:padh//2+h, padw//2:padw//2+w]

            under_illu, under_ref = model(1-pad_img)
            under_ref = 1-under_ref
            under_illu = under_illu[:,:,padh//2:padh//2+h, padw//2:padw//2+w]
            under_ref = under_ref[:,:,padh//2:padh//2+h, padw//2:padw//2+w]

            img_numpy = img.squeeze().permute(1,2,0).cpu().numpy()
            over_ref_numpy = over_ref.squeeze().permute(1,2,0).cpu().numpy()
            under_ref_numpy = under_ref.squeeze().permute(1,2,0).cpu().numpy()
            final_img_numpy = fuse_multi_exposure_images(img_numpy, over_ref_numpy, under_ref_numpy, 1, 1, 1)

            final_name = '%s.png' % (image_name[0].split('.')[0] + '_finalImg')
            final_img = Image.fromarray(np.clip(final_img_numpy * 255, 0, 255).astype("uint8"))
            final_img.save(os.path.join(result_path, final_name))

            over_ref_name = '%s.png' % (image_name[0].split('.')[0] + '_overRef')
            torchvision.utils.save_image(over_ref, os.path.join(result_path, over_ref_name))

            over_illu_name = '%s.png' % (image_name[0].split('.')[0] + '_overIllu')
            torchvision.utils.save_image(over_illu, os.path.join(result_path, over_illu_name))

            under_ref_name = '%s.png' % (image_name[0].split('.')[0] + '_underRef')
            torchvision.utils.save_image(under_ref, os.path.join(result_path, under_ref_name))

            under_illu_name = '%s.png' % (image_name[0].split('.')[0] + '_underIllu')
            torchvision.utils.save_image(under_illu, os.path.join(result_path, under_illu_name))

            ori_img_name = '%s.png' % (image_name[0])
            torchvision.utils.save_image(img, os.path.join(result_path, ori_img_name))


if __name__ == '__main__':
    test()
