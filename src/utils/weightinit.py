import torch
import torch.nn as nn

# mean 0 and Standardabweichung 0.02
def w_initial(m):
    """ initialize convolutional and batch norm layers in generator and discriminator """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)
