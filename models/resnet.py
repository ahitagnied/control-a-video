import torch
from dataclasses import dataclass, field
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

@dataclass
class UpDownBlock(nn.Module):
    """
    applies upsampling to video data using nearest neighbor interpolation, with an optional 
    convolution applied afterward
    """
    channels: int
    scale_factor: float
    use_conv: bool = False
    use_conv_transpose: bool = False
    outchannels: int = None
    padding: int = 1
    name: str = "conv"
    conv: nn.Module = field(init=False, default=None)

    def __post_init__(self):
        nn.Module.__init__(self)
        self.out_channels = self.out_channels or self.channels

        if self.use_conv_transpose:
            raise NotImplementedError
        
        if self.scale_factor < 1: # downsample: use a conv with stride to reduce spatial dimensions.
            stride = int(round(1/self.scale_factor)) # note: for a scale factor of 0.5, stride is 2
            if self.use_conv:
                self.conv = InflatedConv2d(self.channels, self.out_channels, 3, stride=stride, padding=self.padding)
            else:
                raise NotImplementedError
        else: # upsample: use interpolation, then optionally a conv.
            if self.use_conv:
                self.conv = InflatedConv2d(self.channels, self.out_channels, 3, padding=1)
            else: 
                self.conv = None
    
    def forward(self, hidden_states, output_size=None):
        # downsampling: directly apply the conv (which has a stride)
        if self.scale_factor < 1:
            return self.conv(hidden_states) 
        # upsampling: use nearest neighbor interpolation, then optional conv.
        else: 
            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float32)
            elif hidden_states.shape[0] >= 64:
                hidden_states = hidden_states.contiguous()

            if output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, 
                    scale_factor=[1, self.scale_factor, self.scale_factor],
                    mode="nearest"
                )
            else: 
                hidden_states = F.interpolate(
                    hidden_states, 
                    size=output_size,
                    model="nearest"
                )
            
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)
            if self.use_conv and self.conv is not None:
                hidden_states = self.conv(hidden_states)
            
            return hidden_states


        



    



