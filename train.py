import argparse

import os
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model.AIENet2 import AIENet
from dataset import DataSetTrain, DataSetValid
from loss import *
import utils

from IQA_pytorch import SSIM
from utils import PSNR


parser = argparse.ArgumentParser("CVTEnhancer")
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--gpu_id', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=60, help='epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--img_size', type=int, default=(512, 512), help='(h,w)')
parser.add_argument('--img_size', type=int, default=(528, 528), help='(h,w)')
parser.add_argument('--win_size', type=int, default=12, help='')
# parser.add_argument('--down_r', type=int, default=2, help='')
# parser.add_argument('--train_dir', type=str, default='/mnt/data/yut/datasets/enhancement/SICE/sum1')
parser.add_argument('--train_dir', type=str, default='/mnt/data/yut/datasets/LLIE/LOL-v1/our485/low')
# parser.add_argument('--valid_dir', type=str, default='./data/SICE_VALID/')
parser.add_argument('--valid_dir', type=str, default='/mnt/data/yut/datasets/LLIE/LOL-v1/eval15/')
parser.add_argument('--exp_dir', type=str, default='./exp/cvt_enhancer11_12', help='location of the data corpus')
parser.add_argument('--writer', type=bool, default=True, help='SummaryWriter')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# utils.create_exp_dir(args.exp_dir, scripts_to_save=["train.py", "model.py", "loss.py"])
os.makedirs(args.exp_dir, exist_ok=True)
model_path = args.exp_dir + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.exp_dir + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

logger = utils.create_logger(args.exp_dir)
logger.info("args = %s", args)

np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main(writer):

    train_dataset = DataSetTrain(img_dir=args.train_dir, img_size=args.img_size)
    logger.info("The number of train images = %d", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               pin_memory=True, num_workers=args.num_works, shuffle=True)
    
    valid_dataset = DataSetValid(img_dir=args.valid_dir, SubDirImg='low', SubDirGt='high')
    logger.info("The number of valid images = %d", len(valid_dataset))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,
                                               pin_memory=True, num_workers=0, shuffle=False)

    # model = AIENet(window_size=args.win_size, down_r=args.down_r).cuda()
    model = AIENet(window_size=args.win_size).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters/1e6}MB")

    data_term = nn.MSELoss().cuda()
    smooth_term = ColorHuberTVLoss(region=5, sigma=1).cuda()

    n_epoch = args.epochs
    # warmup_epoch = args.warmup_epochs
    n_iter_per_epoch = len(train_loader)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  eps=1e-8, betas=(0.9, 0.999), lr=args.lr, weight_decay=0.05)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    # scheduler = CosineLRScheduler(optimizer,
    #                             t_initial=int(n_epoch*n_iter_per_epoch),
    #                             t_mul=1.,
    #                             lr_min=5e-6,
    #                             warmup_lr_init=5e-7,
    #                             warmup_t=int(warmup_epoch*n_iter_per_epoch),
    #                             cycle_limit=1,
    #                             t_in_epochs=False,)

    compute_psnr = PSNR()
    compute_ssim = SSIM()
    curr_step, curr_loss = 0, 10.0
    max_avg_psnr, max_avg_ssim = 0.0, 0.0
    model.train()
    for epoch in range(n_epoch):
        losses = []
        for iter, low in enumerate(train_loader):
            curr_step += 1
            low = low.cuda()
            illu = model(low)
            data_loss = data_term(low, illu)
            smooth_loss = smooth_term(low, illu)
            loss = data_loss + smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if iter % 10 == 0:
                logger.info('train-epoch %03d %03d/%03d loss: %2.4f', epoch, iter, n_iter_per_epoch, loss)
            if loss.item() < curr_loss:
                curr_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(model_path, 'best_loss_weights.pt'))
            if writer != None:
                with torch.no_grad():
                    writer.add_scalar('fidelity', data_loss.item(), curr_step)
                    writer.add_scalar('smooth', smooth_loss.item(), curr_step)
                    writer.add_scalar('loss', loss.item(), curr_step)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_step)
        
        logger.info('===== train-epoch %03d %f', epoch, np.average(losses))
        # torch.save(model.state_dict(), os.path.join(model_path, 'weights_%d.pt' % epoch))

        if epoch % 1 == 0 and curr_step != 0:
            logger.info('========= validation-epoch %03d =========', epoch)
            model.eval()
            psnr_list = []
            ssim_list = []
            with torch.no_grad():
                for _, (low, gt, file_name) in enumerate(valid_loader):
                    low = low.cuda()
                    gt = gt.cuda()

                    pad_low = utils.pad_image(low, num_down=1, win_size=args.win_size)
                    pad_illu = model(pad_low)
                    illu = utils.invert_pad_image(pad_illu, low)
                    ref = utils.retinex(low, illu)

                    # save ref/illu image
                    ref_name = '%s.png' % (str(epoch) + '_' + file_name[0].split('.')[0] + '_ref')
                    ref_path = os.path.join(image_path, ref_name)
                    torchvision.utils.save_image(ref, ref_path)
                    illu_name = '%s.png' % (str(epoch) + '_' + file_name[0].split('.')[0] + '_illu')
                    illu_path = os.path.join(image_path, illu_name)
                    torchvision.utils.save_image(illu, illu_path)

                    # compute psnr/ssim
                    psnr = compute_psnr(gt, ref)
                    ssim = compute_ssim(gt, ref, as_loss=False)
                    psnr_list.append(psnr.item())
                    ssim_list.append(ssim.item())
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                logger.info(f"avg_psnr:{avg_psnr}, avg_ssim:{avg_ssim}")
                # save model
                if avg_psnr > max_avg_psnr:
                    max_avg_psnr = avg_psnr
                    torch.save(model.state_dict(), os.path.join(model_path, 'best_psnr_weights.pt'))
                if avg_ssim > max_avg_ssim:
                    max_avg_ssim = avg_ssim
                    torch.save(model.state_dict(), os.path.join(model_path, 'best_ssim_weights.pt'))
    logger.info(f"max_psnr:{max_avg_psnr}, max_ssim:{max_avg_ssim}")



if __name__ == '__main__':
    writer = SummaryWriter(args.exp_dir) if args.writer == True else None
    main(writer)
    if args.writer == True:
        writer.close()
