import os
import torch
import csv
from torch.utils.data import Dataset
import numpy as np
import imageio
import augmentations as aug

# img: 0-255
# depth
SMALL_EPS = 1e-6

class NYUTrainset(Dataset):
    def __init__(self, index_file, aug=None, debug=False):
        self.DEBUG = debug
        self.augmentation = aug

        with open(index_file, 'r') as f:
            self.data_path = list(csv.reader(f))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img = np.asarray(imageio.imread(self.data_path[index][0]), dtype=np.float)
        depth = np.asarray(imageio.imread(self.data_path[index][1]), dtype=np.float)
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        if self.DEBUG:
            import cv2
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth.png'), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
            return 1

        img = np.transpose(img, [2, 0, 1]) / 255
        depth = (depth - depth_min) / (depth_max - depth_min + SMALL_EPS)
        depth = depth[np.newaxis, ...]
        sample = (img, depth)
        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return sample


class NYUTestset(Dataset):
    def __init__(self, index_file):
        with open(index_file, 'r') as f:
            self.data_path = list(csv.reader(f))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img = np.asarray(imageio.imread(self.data_path[index][0]), dtype=np.float)
        depth = np.asarray(imageio.imread(self.data_path[index][1]), dtype=np.float)
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        img = np.transpose(img, [2, 0, 1]) / 255
        depth = (depth - depth_min) / (depth_max - depth_min + SMALL_EPS)
        depth = depth[np.newaxis, ...]
        sample = (img, depth)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils import data
    bs = 1
    augs = aug.Compose([aug.RandomHorizontallyFlip()])
    dst = NYUTrainset(index_file='data/train_debug.csv', aug=augs, debug=False)
    dst_test = NYUTestset(index_file='data/train_debug.csv')
    trainloader = data.DataLoader(dst_test, batch_size=1)
    for i, data in enumerate(trainloader):
        img, depth = data
        print(img)
        print(depth)
        img = np.transpose(img, [0, 2, 3, 1])
        f, axarr = plt.subplots(2)
        for j in range(bs):
            axarr[0].imshow(img[j])
            axarr[1].imshow(depth[j][0])
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()