import torch
import torch.nn as nn

from instant_nsr import models
from instant_nsr.models.utils import get_activation
from instant_nsr.models.network_utils import get_mlp


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.network = network
    
    def forward(self, features, dirs, *args):
        network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()

        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
