import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import math

class UpsamplingLayer(nn.Module):
    def __init__(self):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Upsample(scale_factor=2)

    def forward(self, x):
        if isinstance(self.layer, nn.Upsample):
            self.layer.recompute_scale_factor = False
        return self.layer(x)


#####
# Currently default generator we use
# conv0 -> conv1 -> conv2 -> resnet_blocks -> upconv2 -> upconv1 ->  conv_11 -> (conv_11_a)* -> conv_12 -> (Tanh)*
# there are 2 conv layers inside conv_11_a
# * means is optional, model uses skip-connections
class GeneratorJ(nn.Module):
    def __init__(self, norm_layer='batch_norm',
                 gpu_ids=None, use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 64), input_channels=3, append_smoothers=False):
        super(GeneratorJ, self).__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm'], \
            "norm_layer should be None, 'batch_norm' or 'instance_norm', not {}".format(norm_layer)
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.gpu_ids = gpu_ids
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_smoothers = append_smoothers

        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=7, stride=1, padding=3,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.conv1 = self.relu_layer(in_filters=filters[0],
                                     out_filters=filters[1],
                                     size=3, stride=2, padding=1,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.conv2 = self.relu_layer(in_filters=filters[1],
                                     out_filters=filters[2],
                                     size=3, stride=2, padding=1,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[2],
                                  out_filters=filters[2],
                                  size=3, stride=1, padding=1,
                                  bias=self.use_bias,
                                  norm_layer=self.norm_layer,
                                  nonlinearity=nn.ReLU()))

        self.upconv2 = self.upconv_layer_upsample_and_conv(in_filters=filters[3] + filters[2],
                                         # in_filters=filters[3], # disable skip-connections
                                         out_filters=filters[4],
                                         size=4, stride=2, padding=1,
                                         bias=self.use_bias,
                                         norm_layer=self.norm_layer,
                                         nonlinearity=nn.ReLU())

        self.upconv1 = self.upconv_layer_upsample_and_conv(in_filters=filters[4] + filters[1],
                                         # in_filters=filters[4],  # disable skip-connections
                                         out_filters=filters[4],
                                         size=4, stride=2, padding=1,
                                         bias=self.use_bias,
                                         norm_layer=self.norm_layer,
                                         nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_channels=filters[0] + filters[4] + input_channels,
                      # in_channels=filters[4],  # disable skip-connections
                      out_channels=filters[5],
                      kernel_size=7, stride=1, padding=3, bias=self.use_bias),
            nn.ReLU()
        )

        if self.append_smoothers:
            self.conv_11_a = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]), # replace with variable
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU()
            )

        if tanh:
            self.conv_12 = nn.Sequential(nn.Conv2d(filters[5], 3,
                                                   kernel_size=1, stride=1,
                                                   padding=0, bias=True),
                                         nn.Tanh())
        else:
            self.conv_12 = nn.Conv2d(filters[5], 3, kernel_size=1, stride=1,
                                     padding=0, bias=True)

    def forward(self, x):
        output_0 = self.conv0(x) # 40, 5, 32, 32 -> 40, 32, 32, 32
        output_1 = self.conv1(output_0) # 40, 32, 32, 32 -> 40, 64, 16, 16
        output_2 = self.conv2(output_1) # 40, 64, 16, 16 -> 40, 128, 8, 8
        
        output = output_2
        for layer in self.resnets:
            output = layer(output) + output # 40, 128, 8, 8

        output = self.upconv2(torch.cat((output, output_2), dim=1)) # 40, 128+128, 8, 8 -> 40, 128, 16, 16
        output = self.upconv1(torch.cat((output, output_1), dim=1)) # 40, 128+64, 16, 16 -> 40, 128, 32, 32
        output = self.conv_11(torch.cat((output, output_0, x), dim=1)) # 40, 128+32, 32, 32 -> 40, 64, 32, 32

        if self.append_smoothers:
            output = self.conv_11_a(output)
        output = self.conv_12(output) # 40, 64, 32, 32 -> 40, 3, 32, 32
        return output

    def relu_layer(self, in_filters, out_filters, size, stride, padding, bias,
                   norm_layer, nonlinearity):
        out = nn.Sequential()
        out.add_module('conv', nn.Conv2d(in_channels=in_filters,
                                         out_channels=out_filters,
                                         kernel_size=size, stride=stride,
                                         padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity', nonlinearity)
        return out

    def resnet_block(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = nn.Sequential()
        if nonlinearity:
            out.add_module('nonlinearity_0', nonlinearity)
        out.add_module('conv_0', nn.Conv2d(in_channels=in_filters,
                                           out_channels=out_filters,
                                           kernel_size=size, stride=stride,
                                           padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity_1', nonlinearity)
        out.add_module('conv_1', nn.Conv2d(in_channels=in_filters,
                                           out_channels=out_filters,
                                           kernel_size=size, stride=stride,
                                           padding=padding, bias=bias))
        return out

    def upconv_layer(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = nn.Sequential()
        out.add_module('upconv', nn.ConvTranspose2d(in_channels=in_filters,
                                                    out_channels=out_filters,
                                                    kernel_size=size, # 4
                                                    stride=stride, # 2
                                                    padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity', nonlinearity)
        return out

    def upconv_layer_upsample_and_conv(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):

        parts = [UpsamplingLayer(),
                 nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=False)]

        if norm_layer:
            parts.append(norm_layer(num_features=out_filters))

        if nonlinearity:
            parts.append(nonlinearity)

        return nn.Sequential(*parts)
    
    
#####
# Currently default generator we use
# conv0 -> conv1 -> conv2 -> resnet_blocks -> upconv2 -> upconv1 ->  conv_11 -> (conv_11_a)* -> conv_12 -> (Tanh)*
# there are 2 conv layers inside conv_11_a
# * means is optional, model uses skip-connections
class GeneratorJ_RIC(nn.Module):
    def __init__(self, norm_layer='batch_norm',
                 gpu_ids=None, use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 64), input_channels=3, append_smoothers=False):
        super(GeneratorJ_RIC, self).__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm'], \
            "norm_layer should be None, 'batch_norm' or 'instance_norm', not {}".format(norm_layer)
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.gpu_ids = gpu_ids
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_smoothers = append_smoothers

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=3, stride=1, padding=1,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.conv1 = self.relu_layer(in_filters=filters[0],
                                     out_filters=filters[1],
                                     size=3, stride=1, padding=1,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.conv2 = self.relu_layer(in_filters=filters[1],
                                     out_filters=filters[2],
                                     size=3, stride=1, padding=1,
                                     bias=self.use_bias,
                                     norm_layer=self.norm_layer,
                                     nonlinearity=nn.LeakyReLU(.2))

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[2],
                                  out_filters=filters[2],
                                  size=3, stride=1, padding=1,
                                  bias=self.use_bias,
                                  norm_layer=self.norm_layer,
                                  nonlinearity=nn.ReLU()))

        self.upconv2 = self.upconv_layer_upsample_and_conv(in_filters=filters[3] + filters[2],
                                         out_filters=filters[4],
                                         size=3, stride=2, padding=1,
                                         bias=self.use_bias,
                                         norm_layer=self.norm_layer,
                                         nonlinearity=nn.ReLU())

        self.upconv1 = self.upconv_layer_upsample_and_conv(in_filters=filters[4] + filters[1],
                                         out_filters=filters[4],
                                         size=3, stride=2, padding=1,
                                         bias=self.use_bias,
                                         norm_layer=self.norm_layer,
                                         nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_channels=filters[0] + filters[4] + input_channels,
                      out_channels=filters[5],
                      kernel_size=3, stride=1, padding=1, bias=self.use_bias),
            nn.ReLU()
        )

        if self.append_smoothers:
            self.conv_11_a = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]), # replace with variable
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU()
            )

        if tanh:
            self.conv_12 = nn.Sequential(nn.Conv2d(filters[5], 3,
                                                   kernel_size=1, stride=1,
                                                   padding=0, bias=True),
                                         nn.Tanh())
        else:
            self.conv_12 = nn.Conv2d(filters[5], 3, kernel_size=1, stride=1,
                                     padding=0, bias=True)


        self.coords_0, self.coords_1, self.coords_2 = None, None, None
        self.current_x_shape = None

    def forward(self, x):

        # B, 5, 32, 32 -> B, 32, 32, 32
        if x.shape != self.current_x_shape:
            self.current_x_shape = x.shape
            self.coords_0 = generate_coordinates(x.shape[0], x.shape[2], x.shape[3])
            self.coords_1 = generate_coordinates(x.shape[0], int(x.shape[2]/2), int(x.shape[3]/2))
            self.coords_2 = generate_coordinates(x.shape[0], int(x.shape[2]/4), int(x.shape[3]/4))

        tmp = torchvision.ops.deform_conv2d(input=x, offset=self.coords_0, weight=self.conv0[0].weight, padding=(1,1))
        tmp = self.conv0[1](tmp)
        output_0 = self.conv0[2](tmp)

        # B, 32, 32, 32 -> B, 64, 16, 16
        tmp = self.maxpool(output_0)
        tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_1, weight=self.conv1[0].weight, padding=(1,1))
        tmp = self.conv1[1](tmp)
        output_1 = self.conv1[2](tmp)

        # B, 64, 16, 16 -> B, 128, 8, 8
        tmp = self.maxpool(output_1)
        tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_2, weight=self.conv2[0].weight, padding=(1,1))
        tmp = self.conv2[1](tmp)
        output_2 = self.conv2[2](tmp)
        
        # B, 128, 8, 8
        output = output_2
        for layer in self.resnets:
            tmp = layer[0](output)
            tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_2, weight=layer[1].weight, padding=(1,1))
            tmp = layer[2](tmp)
            tmp = layer[3](tmp)
            tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_2, weight=layer[4].weight, padding=(1,1))
            output = tmp + output

        # B, 128+128, 8, 8 -> B, 128, 16, 16
        tmp = torch.cat((output, output_2), dim=1)
        tmp = self.upconv2[0](tmp)
        tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_1, weight=self.upconv2[1].weight, padding=(1,1))
        tmp = self.upconv2[2](tmp)
        output = self.upconv2[3](tmp)

        # B, 128+64, 16, 16 -> B, 128, 32, 32
        tmp = torch.cat((output, output_1), dim=1)
        tmp = self.upconv1[0](tmp)
        tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_0, weight=self.upconv1[1].weight, padding=(1,1))
        tmp = self.upconv1[2](tmp)
        output = self.upconv1[3](tmp)

        # B, 128+32, 32, 32 -> B, 64, 32, 32
        tmp = torch.cat((output, output_0, x), dim=1)
        tmp = torchvision.ops.deform_conv2d(input=tmp, offset=self.coords_0, weight=self.conv_11[0].weight, padding=(1,1))
        output = self.conv_11[1](tmp)

        if self.append_smoothers:
            tmp = torchvision.ops.deform_conv2d(input=output, offset=self.coords_0, weight=self.conv_11_a[0].weight, padding=(1,1))
            tmp = self.conv_11_a[1](tmp)
            tmp = self.conv_11_a[2](tmp)
            tmp = torchvision.ops.deform_conv2d(input=output, offset=self.coords_0, weight=self.conv_11_a[3].weight, padding=(1,1))
            output = self.conv_11_a[4](tmp)

        # B, 64, 32, 32 -> B, 3, 32, 32
        output = self.conv_12(output) 
        return output

    def relu_layer(self, in_filters, out_filters, size, stride, padding, bias,
                   norm_layer, nonlinearity):
        out = nn.Sequential()
        out.add_module('conv', nn.Conv2d(in_channels=in_filters,
                                         out_channels=out_filters,
                                         kernel_size=size, stride=stride,
                                         padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity', nonlinearity)
        return out

    def resnet_block(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = nn.Sequential()
        if nonlinearity:
            out.add_module('nonlinearity_0', nonlinearity)
        out.add_module('conv_0', nn.Conv2d(in_channels=in_filters,
                                           out_channels=out_filters,
                                           kernel_size=size, stride=stride,
                                           padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity_1', nonlinearity)
        out.add_module('conv_1', nn.Conv2d(in_channels=in_filters,
                                           out_channels=out_filters,
                                           kernel_size=size, stride=stride,
                                           padding=padding, bias=bias))
        return out

    def upconv_layer(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = nn.Sequential()
        out.add_module('upconv', nn.ConvTranspose2d(in_channels=in_filters,
                                                    out_channels=out_filters,
                                                    kernel_size=size, # 4
                                                    stride=stride, # 2
                                                    padding=padding, bias=bias))
        if norm_layer:
            out.add_module('normalization',
                           norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity', nonlinearity)
        return out

    def upconv_layer_upsample_and_conv(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):

        parts = [UpsamplingLayer(),
                 nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=False)]

        if norm_layer:
            parts.append(norm_layer(num_features=out_filters))

        if nonlinearity:
            parts.append(nonlinearity)

        return nn.Sequential(*parts)



#####
# Default discriminator
#####
class DiscriminatorN_IN(nn.Module):
    def __init__(self, num_filters=64, input_channels=3, n_layers=3,
                 use_noise=False, noise_sigma=0.2, norm_layer='instance_norm', use_bias=True):
        super(DiscriminatorN_IN, self).__init__()

        self.num_filters = num_filters
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.input_channels = input_channels
        self.use_bias = use_bias

        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.InstanceNorm2d
        self.net = self.make_net(n_layers, self.input_channels, 1, 4, 2, self.use_bias)
    
    def make_net(self, n, flt_in, flt_out=1, k=4, stride=2, bias=True):
        padding = 1
        model = nn.Sequential()

        model.add_module('conv0', self.make_block(flt_in, self.num_filters, k, stride, padding, bias, None, nn.LeakyReLU))

        flt_mult, flt_mult_prev = 1, 1
        # n - 1 blocks
        for l in range(1, n):
            flt_mult_prev = flt_mult
            flt_mult = min(2**(l), 8)
            model.add_module('conv_%d'%(l), self.make_block(self.num_filters * flt_mult_prev, self.num_filters * flt_mult, 
                                                              k, stride, padding, bias, self.norm_layer, nn.LeakyReLU))
            
        flt_mult_prev = flt_mult
        flt_mult = min(2**n, 8)
        model.add_module('conv_%d'%(n), self.make_block(self.num_filters * flt_mult_prev, self.num_filters * flt_mult, 
                                                        k, 1, padding, bias, self.norm_layer, nn.LeakyReLU))
        model.add_module('conv_out', self.make_block(self.num_filters * flt_mult, 1, k, 1, padding, bias, None, None))
        return model

    def make_block(self, flt_in, flt_out, k, stride, padding, bias, norm, relu):
        m = nn.Sequential()
        m.add_module('conv', nn.Conv2d(flt_in, flt_out, k, stride=stride, padding=padding, bias=bias))
        if norm is not None:
            m.add_module('norm', norm(flt_out))
        if relu is not None:
            m.add_module('relu', relu(0.2, True))
        return m

    def forward(self, x):
        return self.net(x), None # 2nd is class?


#####
# Perception VGG19 loss
#####
class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True, path=None):
        super(PerceptualVGG19, self).__init__()
        if path is not None:
            print(f'Loading pre-trained VGG19 model from {path}')
            model = models.vgg19(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 40),
            )
            model.load_state_dict(torch.load(path))
        else:
            model = models.vgg19(pretrained=True)
        model.float()
        model.eval()

        self.model = model
        self.feature_layers = feature_layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None

        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None

        self.use_normalization = use_normalization

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if not self.use_normalization:
            return x

        if self.mean_tensor is None:
            self.mean_tensor = Variable(
                self.mean.view(1, 3, 1, 1).expand(x.size()),
                requires_grad=False)
            self.std_tensor = Variable(
                self.std.view(1, 3, 1, 1).expand(x.size()), requires_grad=False)

        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        features = []

        h = x

        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)

        return None, torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)


def generate_coordinates(batch_size, input_height, input_width):
        coords = torch.zeros(input_height, input_width, 2 * 3 * 3)
        
        parameters = torch.zeros(3)
        parameters[0] = batch_size
        parameters[1] = input_height
        parameters[2] = input_width
        
        center = torch.zeros(2)
        center[0]=torch.sub(torch.div(parameters[1], 2.0), 0.5)
        center[1]=torch.sub(torch.div(parameters[2], 2.0), 0.5)
            
        x_grid = torch.arange(0, parameters[1])
        y_grid = torch.arange(0, parameters[2]) 
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
        
        #coords[:,:,8]=grid_x
        #coords[:,:,9]=grid_y
               
        delta_x = torch.sub(grid_x, center[0])
        delta_y = torch.sub(grid_y, center[1])
        PI = torch.mul(torch.Tensor([math.pi]), 2.0)
        theta=torch.atan2(delta_y, delta_x) % PI[0]
        theta=torch.round(10000.*theta)/10000.
        
        coords[:,:,0]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)
        coords[:,:,1]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)    
        
        coords[:,:,2]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0)
        coords[:,:,3]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),0.0)
        
        coords[:,:,4]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0)
        coords[:,:,5]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),-1.0)
        
        coords[:,:,6]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),0.0)
        coords[:,:,7]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0)
        
        coords[:,:,10]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),0.0)
        coords[:,:,11]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),-1.0)
        
        coords[:,:,12]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),-1.0)
        coords[:,:,13]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0)
        
        coords[:,:,14]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),-1.0)
        coords[:,:,15]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),0.0)
        
        coords[:,:,16]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
        coords[:,:,17]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
        
        coords=coords.expand(batch_size,-1,-1,-1) 
        coords=coords.permute(0, 3, 1, 2)
        coords = coords.cuda()
        
        return Variable(coords, requires_grad=False)

