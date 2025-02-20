import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial # fix certain arguments of a function, to create a higher order function

class PseudoConv3D(nn.Conv2d):
    """
    applies a spatial 2d convolutional + 1d temporal convolution for video data processing 
    """
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size, **kwargs):
        # pass arguments onto the parent class to do all the usual setup for a conv layer
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )

        # if temporal kernel is not present 
        if temporal_kernel_size == None: temporal_kernel_size = kernel_size

        # initialise a temporal 1d conv layer
        self.conv_temporal = (nn.Conv1d(
            out_channels, 
            out_channels,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size//2
        ) if kernel_size > 1 else None)

        # if self.conv_temporal is not none, then initialise its weights to identity and bias to 0
        if self.conv_temporal is not None: 
            nn.init.dirac_(self.conv_temporal.weight.data)
            nn.init.zeros_(self.conv_temporal.bias.data)
        
    def forward(self, x):
        b = x.shape[0] # num of batches
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x) # pass through a convoution layer of the parent class
        x = rearrange(x, "(b f) c h w -> b f c h w") # rearrange to video shape 

        *_, h, w = x.shape # retain values of h and w for rearranging
        
        # now do a temporal convolution 
        x = rearrange(x, "b f c h w -> (b h w) f c")
        x = self.conv_temporal(x)
        x = rearrange(x, "(b h w) f c -> b g c h w", h = h, w = w)

        return x