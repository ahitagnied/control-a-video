import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class InflatedConv2d(nn.Conv2d):
    """
    applies a 2d convolution layer across the frames of a video. each frame is convolved 
    independently
    """
    def forward(self, x):
        # extract the number of frames in the video
        video_length = x.shape[2]
        # (b c f h w) -> ((b*f) f h w)
        x = rearrange(x, "b f c h w -> (b f) c h w")
        # apply 2d convolution
        x = super().conv2d(x)
        x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)
        return x

class TemporalConv1d(nn.Conv2d):
    """
    applies a 1d temporal convolution to video data. each spatial location's time sequence is 
    convolved independently, and then the original shape is restored
    """
    def forward(self, x):
        b, c, f, h, w = x.shape
        # rearrange to y for temporal conv; b c f h w -> (b h w) c f
        y = rearrange(x.clone(), "b c f h w -> (b h w) c f") 
        y = super().forward(y)
        # rearrange back to b, c, f, h, w
        y = rearrange(y, "(b h w) c f -> b c f h w", b=b, h=h, w=w)
        return y

