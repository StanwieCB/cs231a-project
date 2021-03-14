"""
This module enable using str to load a model class
"""

import models


# TODO: just an example
class Autoencoder(models.Autoencoder):
    def __init__(self):
        super(Autoencoder, self).__init__(input_channels=3)

class UNet(models.UNet):
    def __init__(self):
        super(UNet, self).__init__(n_channels=3, n_classes=1)


