import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial # fix certain arguments of a function, to create a higher order function

class PseudoConv3D(nn.Conv2d):
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

    does: input video: x -> F.interpolate to increase h/w -> optinally use a PseudoConv3D for spatial + temporal convolution
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
            conv = PseudoConv3D(self.channels, self.out_channels, 3, padding=1)
        
        # store convolution layer
        if name == "conv": # use name for compatibility issues 
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
        self.out_channels=out_channels or channels # if output channels are not specified -> same as input
        self.padding=padding
        stride=2
        self.name=name

        if use_conv:
            # downsample using strided convolution
            conv = PseudoConv3D(self.channels, self.out_channels, kernel_size=3,stride=stride, padding=padding)
        else: 
            # downsample using average pooling if no convolution needed
            # ensures input and output channels match when pooling
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.conv2d0 = conv
        self.conv = conv
    
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels # check if input has correct number of channels 
        # adds one column of zeros on right and one row of zeros on bottom when no padding specified
        if self.use_conv and self.padding == 0:
            padding = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, padding, mode="constant", value=0)
        # optinally add convolution
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        else: 
            # use avgpool instead
            b = hidden_states.shape[0] # get number of batches
            is_video = hidden_states.ndim == 5 # check if it is a video
            if is_video: 
                # same process as in upsample3d
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            hidden_states = self.conv(hidden_states) # apply avg pool
            if is_video: 
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w") # back to video dimensions

        return hidden_states

class ResnetBlockPseudo3D(nn.Module): 
    """
    
    """
    def __init__(
            self, 
            *,
            in_channels, 
            out_channels=None,
            conv_shortcut=False, # whether to use convolution in skip connection
            dropout=0.0, # dropout probability
            temb_channels=512, # number of time embedding channels
            groups=32, # number of groups for group normalization
            groups_out=None, # number of output groups (defaults to groups if none)
            pre_norm=True, # whether to normalize before convolution
            eps=1e-6, # small value for numerical stability
            non_linearity="swish", # activation function type
            time_embedding_norm="default", # how to handle time embeddings
            kernel=None, # kernel type for up/downsampling
            output_scale_factor=1.0, # scale factor for outputs
            use_in_shortcut=None, # whether to use input in skip connection
            up=False, # whether to upsample
            down=False, # whether to downsample
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        # set output groups same as input if not specified
        if groups_out == None:
            groups_out = groups

        # first normalisation layer
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # first convolutional layer
        self.conv1 = PseudoConv3D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # time embedding projection if needed
        if temb_channels is not None:
            # two ways to apply time information:
            # 1. default: adds time info directly to features
            # output channels same as feature channels
            if self.time_embedding_norm == "default": time_emb_proj_out_channels = out_channels
            # 2. scale-shift: uses time to both multiply and add to features
            # needs double channels - half for scaling, half for shifting
            elif self.time_embedding_norm == "scale_shift": time_emb_proj_out_channels = out_channels * 2
            else: raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm}")
            # create linear layer to project time embedding from temb_channels (e.g. 512) to required output size
            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else: self.time_emb_proj = None # no time conditioning

        # second normalization layer
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        # drop out layer
        self.dropout = nn.Dropout(dropout)

        # second convolutional layer
        self.conv2 = PseudoConv3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # set activation function
        if non_linearity=="swish":
            self.non_linearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.non_linearity = lambda x: F.Mish(x)
        elif non_linearity == "silu":
            self.non_linearity = nn.SiLU()

        # initialize up/downsampling to none
        self.upsample = self.downsample = None

        # set up upsampling if required
        if self.up:
            if kernel == 'fir': 
                # finite impulse response kernel
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                # nearest neighbor interpolation
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                # default pseudo3d upsampling
                self.upsample = UpsamplePseudo3D(in_channels, use_conv=False)
        
        # setup downsampling if requested
        elif self.down:
            if kernel == "fir":
                # finite impulse response kernel
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                # average pooling
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                # default pseudo3d downsampling
                self.downsample = DownsamplePseudo3D(in_channels, use_conv=False, padding=1, name="op")

        # determine if input shortcut is needed
        self.use_in_shortcut = (self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut)

        # setup shortcut convolution if needed
        self.conv_shortcut = None
        if self.use_in_shortcut:
            # 1x1 convolution to match channel dimensions
            self.conv_shortcut = PseudoConv3D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """
    upsamples 2d images using a filter kernel
    returns: upsampled tensor [n, c, h * factor, w * factor]
    """
    # verify that factor is an integer and at least 1
    assert isinstance(factor, int) and factor >= 1

    # if no kernel provided, create default kernel of [1,1] for factor=2 (nearest neighbor upsampling)
    if kernel is None: kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)

    # if kernel is 1D, make it 2D by computing outer product with itself
    if kernel.ndim == 1: kernel = torch.outer(kernel, kernel)

    # normalize kernel so sum of all elements = 1
    kernel /= torch.sum(kernel)

    # scale kernel by gain and factor squared
    kernel = kernel * (gain * (factor**2))

    # calculate padding needed based on kernel size and upsampling factor
    pad_value = kernel.shape[0] - factor

    # perform upsampling using native implementation:
    # - moves kernel to same device as input
    # - specifies upsampling factor
    # - applies calculated padding
    output = upfirdn2d(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )

    # return upsampled tensor with shape [N, C, H * factor, W * factor]
    return output

def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """
    upsamples 2d images using a filter kernel
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None: kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1: kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output


def upfirdn2d(tensor, kernel, up=1, down=1, pad=(0, 0)):
    """
    performs upsampling, filtering, and downsampling on 2d tensors
    """
    batch, channel, in_h, in_w = tensor.shape
    kernel_h, kernel_w = kernel.shape

    # upsample by inserting zeros
    if up > 1:
        tensor_reshaped = tensor.view(batch, channel, in_h, 1, in_w, 1)
        tensor = F.pad(tensor_reshaped, [0, up - 1, 0, 0, 0, up - 1, 0, 0])
        tensor = tensor.view(batch, channel, in_h * up, in_w * up)

    # padding
    pad_x0, pad_x1 = max(pad[0], 0), max(pad[1], 0)
    pad_y0, pad_y1 = max(pad[0], 0), max(pad[1], 0)
    if any((pad_x0, pad_x1, pad_y0, pad_y1)):
        tensor = F.pad(tensor, [pad_x0, pad_x1, pad_y0, pad_y1])

    # crop if negative padding
    if pad[0] < 0 or pad[1] < 0:
        tensor = tensor[:, :,
                      max(-pad[0], 0):tensor.shape[2] - max(-pad[1], 0),
                      max(-pad[0], 0):tensor.shape[3] - max(-pad[1], 0)]

    # apply filter
    kernel_tensor = torch.flip(kernel, [0, 1]).unsqueeze(0).unsqueeze(0)
    kernel_tensor = kernel_tensor.repeat(channel, 1, 1, 1)
    tensor = F.conv2d(tensor, kernel_tensor, groups=channel)

    # downsample
    if down > 1:
        tensor = tensor[:, :, ::down, ::down]

    return tensor