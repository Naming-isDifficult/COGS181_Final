import torch
import torch.nn as nn

'''
  A general layer for contracting path of the U-net
  It contains two Convolutional layers with kernel 3x3
and one max-pooling with kernel 2x2
  No paddings will be applied to the input ---- all paddings
should be done before feeding the network
  The input of this layer is assumed to have shape
      [batch, input_channel, height, width]
  The output of this layer has shape
      [batch, output_channel, (height-4)/2, (width-4)/2]
'''
class Down(nn.Module):
    
    def __init__(self, input_channel, output_channel, has_bn=False):
        super(Down, self).__init__()

      #the first convolution layer
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels = input_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'valid')
                      
            nn.ReLU()

        )
        #apply batch normalization if needed
        #by default there should not be one
        if has_bn:
            self.conv1.add_module("batch_norm",
                nn.BatchNorm2d(output_channel)
            )

      #the second convolution layer
        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels = output_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'valid')
                      
            nn.ReLU()

        )
        #apply batch normalization if needed
        #by default there should not be one
        if has_bn:
            self.conv2.add_module("batch_norm",
                nn.BatchNorm2d(output_channel)
            )

      #pooling
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out


'''
  A general layer for expanding path of the U-net.
  It contains two Convolutional layers with kernel 3x3
and one up-conv layer with kernel 2x2.
  If the parameter last_layer is True, the up-conv layer
will be suppressed to fulfil the output of the last layer.
  The input of this layer is assumed to have shape
      [batch, input_channel, height, width]
  The output of this layer has shape
      [batch, output_channel, (height-4)*2, (width-4)*2]
'''
class Up(nn.Module):
    
    def __init__(self, input_channel, output_channel,\
                 has_bn=False, last_layer=False):
        super(Down, self).__init__()

      #the first convolution layer
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels = input_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'valid')
                      
            nn.ReLU()

        )
        #apply batch normalization if needed
        #by default there should not be one
        if has_bn:
            self.conv1.add_module("batch_norm",
                nn.BatchNorm2d(output_channel)
            )

      #the second convolution layer
        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels = output_channel,\
                      out_channels = output_channel,\
                      kernel_size = (3,3),\
                      padding = 'valid')
                      
            nn.ReLU()

        )
        #apply batch normalization if needed
        #by default there should not be one
        if has_bn:
            self.conv2.add_module("batch_norm",
                nn.BatchNorm2d(output_channel)
            )

      #up-conv
        self.up_conv = None
        if not last_layer:
            self.up_conv = nn.ConvTranspose2d(\
                              in_channels = output_channel,\
                              out_channels = output_channel,\
                              kernel_size = (2,2),\
                              stride = 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out