import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial # fix certain arguments of a function, to create a higher order function

class PseudoConv3d(nn.Conv2d):
    """
    applies a spatial 2d convolutional + 1d temporal convolution for video data processing 
    """
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size=None, **kwargs):
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
        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x) # pass through a convoution layer of the parent class
        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b) # rearrange to video shape 

        *_, h, w = x.shape # retain values of h and w for rearranging
        
        if self.conv_temporal is None or not is_video:
            return x
        
        # now do a temporal convolution 
        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.conv_temporal(x)
        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x

class UpsamplePseudo3D(nn.Module):
    """
    spatial upsampling layer with optional convolution
    """
    def __init__(self, channels, use_conv=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels=channels # number of input channels 
        self.use_conv=use_conv # whether to use convolution or not
        self.out_channels=out_channels or channels # number of output channels, default to input channels 
        self.name=name # name used to store convolution data

        # initialise convolution layer based on settings
        conv = None
        if use_conv: # use PseudoConv3d for spatial+temporal convolution
            conv = PseudoConv3d(self.channels, self.out_channels, 3, padding=1)
        
        # store convolution layer
        if name == "conv":
            self.conv = conv
        else: 
            self.Conv2d0 = conv
    
    def forward(self, hidden_states, output_size=None):
        # verify that input has correct number of channels 
        assert hidden_states.shape[1] == self.channels

        # change dtype for compatibility issues
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.float32)
        
        # if the batch size is large, then make sure hidden_states is in a contiguous block of memory
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # check if this is a video
        is_video = hidden_states.ndim == 5
        b = hidden_states.shape[0] # find the number of batches
        if is_video: # video shape -> image shape 
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        if output_size is None: 
            # repeats every weight twice in height and width directions, doubling the size
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else: 
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        
        # if the input is dfloat16, switch back to dfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        # image shape -> video shape
        if is_video:
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=b)

        # optional convolution
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else: 
                hidden_states = self.Conv2d0(hidden_states)
        
        return hidden_states
    

class DownsamplePseudo3D(nn.Module):
    """
    spatial downsampling layer with optional convolution
    """
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels=channels 
        self.use_conv=use_conv
        self.out_channels=out_channels or channels
        self.padding=padding
        stride=2
        self.name=name

        if use_conv:
            conv = PseudoConv3d(self.channels, self.out_channels, kernel_size=3,stride=stride, padding=padding)
        else: 
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.conv2d0 = conv
        self.conv = conv
    
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            padding = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, padding, mode="constant", value=0)