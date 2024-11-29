import numpy as np
import torch
import torch.nn as nn

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1) 

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    
class HierarchicalPE(nn.Module):
    def __init__(self, channels, depth):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(HierarchicalPE, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)
        self.depth = depth
    def forward(self, tensor, depth):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 2:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        x, orig_ch = tensor.shape
        pos_x = torch.zeros(x, device=tensor.device).type(self.inv_freq.type())
        s = 0
        for i in range(depth):
            m = s + 2**(depth-i-1)
            e = s + 2**(depth-i)
            pos_x[s:m] = 2**(depth-i-1) + torch.arange(2**(depth-i-1), device=tensor.device).type(self.inv_freq.type())
            pos_x[m:e] = -1*2**(depth-i-1) - torch.arange(2**(depth-i-1), device=tensor.device).type(self.inv_freq.type())
            s = e
        # pos_x[:16] = 16+ torch.arange(16, device=tensor.device).type(self.inv_freq.type())
        # pos_x[16:32] = -16 - torch.arange(16).type(self.inv_freq.type())
        # pos_x[32:40] = 8 + torch.arange(8, device=tensor.device).type(self.inv_freq.type())
        # pos_x[40:48] = -8 - torch.arange(8, device=tensor.device).type(self.inv_freq.type())
        # pos_x[48:52] = 4 + torch.arange(4, device=tensor.device).type(self.inv_freq.type())
        # pos_x[52:56] = -4 - torch.arange(4, device=tensor.device).type(self.inv_freq.type())
        # pos_x[56:58] = 2 + torch.arange(2, device=tensor.device).type(self.inv_freq.type())
        # pos_x[58:60] = -2 - torch.arange(2, device=tensor.device).type(self.inv_freq.type())
        # pos_x[60:61] = 1 + torch.arange(1, device=tensor.device).type(self.inv_freq.type())
        # pos_x[61:62] = -1 - torch.arange(1, device=tensor.device).type(self.inv_freq.type())
        # # pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[:, :orig_ch]
        return self.cached_penc
    