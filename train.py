import os
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import filesys
import selector
import global_var
import time
import augmentations as aug

from loss import ssim as ssim_criterion
from loss import depth_loss as gradient_criterion
from dataset import NYUTrainset, NYUTestset
from misc import AverageMeter

class Trainer(object):
    def __init__(self, params):
        self.device = torch.device("cuda:0" if params['device'] == "cuda" else "cpu")
        self.bs = params['batch_size']
        self.log_dir = filesys.prepare_log_dir(params['log_name'])
        filesys.save_params(self.log_dir, params, save_name='params')
        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']
        augs = aug.Compose([aug.RandomHorizontallyFlip()])
        train_dataset = NYUTrainset(index_file=params['train_idx'], aug=augs)
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, drop_last=True if len(train_dataset) > self.bs else False, shuffle=True)
        test_dataset = NYUTestset(index_file=params['test_idx'])
        self.testset_len = len(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)
        self.L1_criterion = nn.L1Loss()
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir, 'tensorboard', time.asctime(time.localtime(time.time()))))
        self.model_name = params['model_name']
        self.model = getattr(selector, self.model_name)()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, '{}.pth.tar').format(self.model_name))
            self.model.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
            self.optimizer.load_state_dict(state_dict)

        self.alpha = params['alpha']
        self.beta = params['beta']
        self.theta = params['theta']
        self.val_best_loss = 1e5

    def train(self, epoch):
        """
        Train for an epoch
        """
        epoch_loss = AverageMeter()
        self.model.train()
        for i, data in enumerate(self.train_loader):
            # Unpack 
            input_img = data[0].to(self.device, dtype=torch.float)
            depth_gt = data[1].to(self.device, dtype=torch.float)

            # Step 
            self.optimizer.zero_grad()
            depth_pred = self.model(input_img)

            l1_loss = self.L1_criterion(depth_pred, depth_gt)
            ssim_loss = torch.clamp((1-ssim_criterion(depth_pred, depth_gt))*0.5, 0, 1)
            grad_loss = gradient_criterion(depth_gt, depth_pred, self.device)

            total_loss = self.alpha * l1_loss + self.beta * ssim_loss + self.theta * grad_loss
            total_loss /= (self.alpha + self.beta + self.theta)
            total_loss.backward()
            self.optimizer.step()
            epoch_loss.update(total_loss.item(), self.bs)
            self.logger.add_scalar("train/loss_l1", l1_loss.item(), self.iter_nums)
            self.logger.add_scalar("train/loss_ssim", ssim_loss.item(), self.iter_nums)
            self.logger.add_scalar("train/loss_grad", grad_loss.item(), self.iter_nums)
            self.logger.add_scalar("train/loss_total", total_loss.item(), self.iter_nums)

            print("Iter {}/{}, loss: {:.4f}".format(self.iter_nums, len(self.train_loader), total_loss.item()))
            self.iter_nums += 1

        self.logger.add_scalar("train_epoch/loss_total", epoch_loss.avg, epoch)
        self._save_ckpt(epoch+1)

    def validate(self, epoch):
        val_final_loss = AverageMeter()
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.test_loader):
                # Unpack 
                input_img = data[0].to(self.device, dtype=torch.float)
                depth_gt = data[1].to(self.device, dtype=torch.float)

                # Step 
                depth_pred = self.model(input_img)

                l1_loss = self.L1_criterion(depth_pred, depth_gt)
                ssim_loss = torch.clamp((1-ssim_criterion(depth_pred, depth_gt))*0.5, 0, 1)
                grad_loss = gradient_criterion(depth_gt, depth_pred, self.device)

                total_loss = self.alpha * l1_loss + self.beta * ssim_loss + self.theta * grad_loss
                total_loss /= (self.alpha + self.beta + self.theta)
                val_final_loss.update(total_loss.item(), self.bs)

                print("Iter {}/{}, loss: {:.4f}".format(i, len(self.test_loader), total_loss.item()))

                # log
                if i % 20 == 0:
                    self.logger.add_scalar('val/loss_total', total_loss.item(), epoch * self.testset_len + i)
                    self.logger.add_image('val/{}/depth_pred'.format(i), torch.clamp(torch.pow(depth_pred.cpu()[0], 0.454545), 0, 1), epoch * self.testset_len + i)
                    self.logger.add_image('val/{}/depth_gt'.format(i), torch.clamp(torch.pow(depth_gt.cpu()[0], 0.454545), 0, 1), epoch * self.testset_len + i)

            if val_final_loss.avg < self.val_best_loss:
                self.val_best_loss = val_final_loss.avg
                self._save_ckpt(epoch, is_val=True)


    def _save_ckpt(self, epoch, is_val=False):
        if is_val:
            save_dir = os.path.join(self.log_dir, "best_val")
        else:
            save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, '{}.pth.tar').format(self.model_name))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_config', default='')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model_name', default="UNet")
    parser.add_argument('--checkpoint', default="")
    parser.add_argument('--log_name', default="UNet-0228-lr1e-4-bs4")
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--train_idx', default='data/nyu2_train.csv')
    parser.add_argument('--test_idx', default='data/nyu2_test.csv')
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--iter_nums', default=0, type=int)
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--theta', default=2.0, type=float)
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
    start_epoch = params['epoch']
    with torch.cuda.device(1):
        trainer = Trainer(params)
        for i in range(100-start_epoch):
            print("epoch: {}".format(i+start_epoch))
            trainer.train(i+start_epoch)
            trainer.validate(i+start_epoch)