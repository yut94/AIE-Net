import numpy as np
import cv2
from glob import glob
import os
from collections import OrderedDict
from natsort import natsort
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

en_paths = natsort.natsorted(glob('demo/output/LOL/*.png'))
gt_paths = natsort.natsorted(glob("/mnt/data/yut/datasets/LLIE/LOL-v1/eval15/high/*.png"))

results = []
for en_path, gt_path in zip(en_paths, gt_paths):

    en = cv2.imread(en_path)[:, :, [2, 1, 0]]
    gt = cv2.imread(gt_path)[:, :, [2, 1, 0]]
    
    result = OrderedDict()
    result['psnr'] = compute_psnr(gt, en)
    result['ssim'], _ = compute_ssim(gt, en, full=True, multichannel=True)

    print(os.path.basename(en_path), "<==>", os.path.basename(gt_path), 
          "psnr:", result['psnr'], "ssim:", result['ssim'])
    results.append(result)

mean_psnr = np.mean([result['psnr'] for result in results])
mean_ssim = np.mean([result['ssim'] for result in results])

print("mean psnr:", mean_psnr)
print("mean_ssim", mean_ssim)



