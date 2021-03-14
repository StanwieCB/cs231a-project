import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, items):
        #print('{} {}'.format(img.shape, label.shape))
        #img, label = Image.fromarray(img, mode='F'), Image.fromarray(label, mode='F')   
        items = list(items)         
        for a in self.augmentations:
            items = a(items)
        return tuple(items)

class RandomHorizontallyFlip(object):
    def __call__(self, items):
        if random.random() < 0.5:
            items = [np.fliplr(item).copy() for item in items]
        return items