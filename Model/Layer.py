import torch
import torch.nn as nn

def get_conv_module(input_channel, output_channel, has_bn=False):
    if has_bn:
        return nn.Sequential(
            nn.Conv2d(in_channels = input_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'same',\
                      padding_mode = 'reflect'),    
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels = input_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'same',\
                      padding_mode = 'reflect'),    
            nn.ReLU()
        )

'''
A general layer for contracting path of the U-net
It contains two Convolutional layers with kernel 3x3 and one max-pooling
with kernel 2x2
No paddings will be applied to the input ---- all paddings should be done
before feeding the network
The input of this layer is assumed to have shape
    [batch, input_channel, height, width]
The output of this layer should contains two values:
    output with shape:
        [batch, output_channel, (height-4)/2, (width-4)/2]
    skip connection with shape:
        [batch, output_channel, (height-4), (width-4)]
'''
class Down(nn.Module):
    
    def __init__(self, input_channel, output_channel,\
                 num_layers = 2, has_bn = False):
        super(Down, self).__init__()

        #initialize convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(get_conv_module(input_channel, output_channel,\
                                                has_bn)) #input layer
        for i in range(num_layers-1):
            self.conv_layers.append(get_conv_module(output_channel,\
                                                    output_channel,\
                                                    has_bn)) #intermediate layer

        #pooling
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        #using the output of first relu as skip
        skip = self.conv_layers[0](x)
        
        #compute output
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.maxpool(out)

        return out, skip


'''
A general layer for expanding path of the U-net.
Unlike original U-net, in order to make sure the output and input have the same
size (it's an auto-encoder), padings will be applied to all regular convolutional
layers except for up-conv layer.
It contains two Convolutional layers with kernel 3x3 and one up-conv layer
with kernel 2x2.
Parameter intermediate_channel stands for the output channel of intermediate
convolution layers.
By default they should have following relationship:
    4*output_channel = 2*intermediate_channel = input_channel
If the parameter last_layer is True, the up-conv layer will be suppressed
to fulfil the output of the last layer.
The input of this layer is assumed to have shape
    [batch, input_channel, height, width]
The output of this layer has shape
    [batch, output_channel, height*2, width*2]
'''
class Up(nn.Module):
    
    def __init__(self, input_channel,\
                 intermediate_channel,\
                 output_channel,\
                 num_layers=2, has_bn=False,\
                 last_layer=False):
        super(Up, self).__init__()

        #initialize convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(get_conv_module(input_channel,\
                                                intermediate_channel,\
                                                has_bn)) #input layer
        for i in range(num_layers-1):
            #output of second last layer should be determined later
            self.conv_layers.append(get_conv_module(intermediate_channel,\
                                                    intermediate_channel,\
                                                    has_bn)) #intermediate layer

        #up-conv
        self.up = None
        if not last_layer:
            self.up = nn.ConvTranspose2d(\
                          in_channels = intermediate_channel,\
                          out_channels = output_channel,\
                          kernel_size = (2,2),\
                          stride = 2)

    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)

        if self.up is not None:
            out = self.up(out)

        return out

'''
A general layer for latent space.
It replace the bottleneck layer in U-net and will perform AdaIN on the input.
Parameter intermediate_channel stands for the output channel of intermediate
convolution layers.
By default they should have following relationship:
    2*input_channel = intermediate_channel
Unlike original U-net, in order to make sure the output and input have the same
size (it's an auto-encoder), padings will be applied to all regular convolutional
layers except for up-conv layer.
It contains two Convolutional layers with kernel 3x3 and one up-conv layer
with kernel 2x2.
The input of this layer is assumed to tuple of two tensors and both of them
are assumed to have shape:
    [batch, input_channel, height, width]
The tuple is assumed to be:
    (content_feature_map, style_feature_map)
The output of this layer should be a tuple:
    (layer_output, transformed_map)
The output of this layer has shape
    [batch, input_channel, height*2, width*2]
The transformed map has shape:
    [batch, input_channel, height, width]
'''
class AdaIN(nn.Module):

    def __init__(self, input_channel, intermediate_channel,\
               has_bn=False):
        super(AdaIN, self).__init__()

        #the first convolution layer
        self.conv1 = get_conv_module(input_channel,\
                                     intermediate_channel,\
                                     has_bn)

        #the second convolution layer
        self.conv2 = get_conv_module(intermediate_channel,\
                                     intermediate_channel,\
                                     has_bn)

        #up-conv
        self.up_conv = nn.ConvTranspose2d(\
                          in_channels = intermediate_channel,\
                          out_channels = input_channel,\
                          kernel_size = (2,2),\
                          stride = 2) #output_channel=input_channel

    #helper method to calculate instance mean
    #assuming x.shape = [batch, channel, height, width]
    #output should have shape [batch, channel, 1, 1]
    def calc_mean(self, x):
        return torch.mean(x, dim=[2,3], keepdim=True)

    #helper method to calculate instance standard deviation
    #assuming x.shape = [batch, channel, height, width]
    #output should have shape [batch, channel, 1, 1]
    def calc_std(self, x):
        return torch.std(x, unbiased = False, dim=[2,3], keepdim=True) + 1e-6
                #add an extremely small value to avoid nan
    
    #freeze encoder (conv_1 should be the last layer of encoder)
    def freeze_encoder(self):
        self.conv1.requires_grad_(requires_grad=False)

    #helper method to perform AdaIN on input
    #assuming content.shape = style.shape = [batch, channel, height, width]
    #output should have shape [batch, channel, height, width]
    def adain(self, content, style):
        content_mean = self.calc_mean(content)
        content_sd = self.calc_std(content)
        style_mean = self.calc_mean(style)
        style_sd = self.calc_std(style)

        #perform adain
        out = style_sd * (content - content_mean) / content_sd\
                + style_mean

        return out

    def forward(self, x):
        #unpack x
        content, style = x
        
        #last layer of encoder
        content = self.conv1(content)
        style = self.conv1(style)

        #adain
        latent = self.adain(content, style)

        #conv
        out = self.conv2(latent)
        out = self.up_conv(out)

        return out, latent