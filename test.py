import os
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import filesys
import selector
import global_var
import time
import cv2
import matplotlib.pyplot as plt

from loss import ssim as ssim_criterion
from loss import depth_loss as gradient_criterion
from dataset import NYUTrainset, NYUTestset
from misc import AverageMeter

class Trainer(object):
    def __init__(self, params):
        self.device = torch.device("cuda:1" if params['device'] == "cuda" else "cpu")
        self.bs = params['batch_size']
        self.log_dir = filesys.prepare_log_dir(params['log_name'])
        filesys.save_params(self.log_dir, params, save_name='params')
        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']
        test_dataset = NYUTestset(index_file=params['test_idx'])
        self.testset_len = len(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)
        self.criterion = nn.MSELoss()
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir, 'tensorboard', time.asctime(time.localtime(time.time()))))
        self.model_name = params['model_name']
        self.model = getattr(selector, self.model_name)()
        self.model.to(self.device)
        ckpt_path = params['checkpoint']
        print('loading ckpt from {}'.format(ckpt_path))
        state_dict = torch.load(os.path.join(ckpt_path, '{}.pth.tar').format(self.model_name))
        self.model.load_state_dict(state_dict)
        self.save_image = params['save_image']

    def test(self):
        val_mse = AverageMeter()
        val_ssim = AverageMeter()
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.test_loader):
                # Unpack 
                input_img = data[0].to(self.device, dtype=torch.float)
                depth_gt = data[1].to(self.device, dtype=torch.float)

                # Step 
                depth_pred = self.model(input_img)

                MSE = self.criterion(depth_pred, depth_gt)
                SSIM = ssim_criterion(depth_pred, depth_gt)
                val_mse.update(MSE.item(), self.bs)
                val_ssim.update(SSIM.item(), self.bs)
                
                if (self.save_image):
                    if not os.path.exists(os.path.join(self.log_dir, 'results')):
                        os.makedirs(os.path.join(self.log_dir, 'results'))
                    save_image(input_img[0].cpu(), '{}/results/color_{}.png'.format(self.log_dir, i))
                    save_image(depth_gt[0].cpu(), '{}/results/gt_{}.png'.format(self.log_dir, i))
                    save_image(depth_pred[0].cpu(), '{}/results/predict_{}.png'.format(self.log_dir, i))

                    image = cv2.imread('{}/results/gt_{}.png'.format(self.log_dir, i), 0)
                    colormap = plt.get_cmap('inferno')
                    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/results/gt_{}.png'.format(self.log_dir, i), heatmap)

                    image = cv2.imread('{}/results/predict_{}.png'.format(self.log_dir, i), 0)
                    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/results/predict_{}.png'.format(self.log_dir, i), heatmap)

                    

                print('Testing: {}'.format(i))
                # # log
                # if i % 20 == 0:
                #     self.logger.add_image('val/{}/color'.format(i), torch.clamp(torch.pow(input_img.cpu()[0], 0.454545), 0, 1), i)
                #     self.logger.add_image('val/{}/depth_pred'.format(i), torch.clamp(torch.pow(depth_pred.cpu()[0], 0.454545), 0, 1), i)
                #     self.logger.add_image('val/{}/depth_gt'.format(i), torch.clamp(torch.pow(depth_gt.cpu()[0], 0.454545), 0, 1), i)

            print('avg MSE: {}'.format(val_mse.avg))
            print('avg SSIM: {}'.format(val_ssim.avg))


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_config', default='')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_name', default="Autoencoder")
    parser.add_argument('--checkpoint', default="log/Autoencoder-0227-lr1e-4-bs4/best_val")
    parser.add_argument('--log_name', default="Autoencoder-0227-test")
    parser.add_argument('--test_idx', default='data/nyu2_test.csv')
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--theta', default=2.0, type=float)
    parser.add_argument('--save_image', default=False, type=bool)
    args = parser.parse_args()

    params = args.__dict__
    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v
    return params


if __name__ == '__main__':
    params = parse_argument()
    with torch.cuda.device(1):
        trainer = Trainer(params)
        trainer.test()