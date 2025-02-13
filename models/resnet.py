import torch
from dataclasses import dataclass, field
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class InflatedConv3d(nn.Conv2d):
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
                self.conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=self.padding)
            else:
                raise NotImplementedError
        else: # upsample: use interpolation, then optionally a conv.
            if self.use_conv:
                self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)
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

@dataclass
class ResnetBlock3D(nn.Module):
    in_channels: int
    out_channels: int = None
    conv_shortcut: bool = False
    dropout: float = 0.0
    temb_channels: int = 512
    groups: int= 32
    groups_out: int = None
    pre_norm: bool= True
    eps: float = 1e-6
    non_linearity: str = "swish"
    time_embedding_norm: str = "default"
    output_scale_factor: float = 1.0
    use_in_shortcut: bool = None

    # the following is initialised by __post_init__
    norm1: nn.Module = field(init=False)
    conv1: nn.Module = field(init=False)
    time_emb_proj: nn.Module = field(init=False)
    norm2: nn.Module = field(init=False)
    dropout_layer: nn.Module = field(init=False)
    conv2: nn.Module = field(init=False)
    nonlinearity_fn: callable = field(init=False)
    conv_shortcut_layer: nn.Module = field(init=False, default=None)

    def __post_init__(self):
        nn.Module.__init__(self)

        # if out_channels is not provided, use in_channels
        if self.out_channels == None:
            self.out_channels = self.in_channels

        # if groups_out is not provided, use groups
        if self.groups_out == None: 
            self.groups_out = self.groups
        
        # as in the original code, force prenorm to be true
        self.pre_norm = True

        # first normalisation
        self.norm1 = nn.GroupNorm(num_groups=self.groups, num_channels=self.in_channels, eps=self.eps, affine=True)
        # first convolution layer: kernel_size=3, stride=1, padding=1
        self.conv1 = InflatedConv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

        # time embedding projection layer (if temb_channels is provided)
        if self.temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = self.out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = self.out_channels * 2
            else: 
                raise ValueError(f"unknown time_embedding_norm: {self.time_embedding_norm}")
            self.time_emb_proj = nn.Linear(self.temb_channels, time_emb_proj_out_channels)
        else: 
            self.time_emb_proj = None

        # second normalisation layer
        self.norm2 = nn.GroupNorm(num_groups=self.groups_out, num_channels=self.out_channels, eps=self.eps, affine=True)
        # dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # second convolution layer: kernel_size=3, stride=1, padding=1
        self.conv2 = InflatedConv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # define the non-linearity
        if self.non_linearity == "swish":
            self.nonlinearity_fn = lambda x: F.silu(x)
        elif self.non_linearity == "mish":
            self.nonlinearity_fn = Mish()
        elif self.non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            raise ValueError(f"unknown non_linearity: {self.non_linearity}")
        
        ## CODE CORRECTION: CHECK!!
        if self.use_in_shortcut is None:
            self.use_in_shortcut = (self.in_channels != self.out_channels)
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

class Mish(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(nn.functional.softplus(hidden_states))