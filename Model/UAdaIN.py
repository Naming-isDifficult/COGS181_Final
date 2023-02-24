from Model.Layer import Down, AdaIN, Up
import torch
import torch.nn as nn

'''
The model. I name it UAdaIN, though I'm bad at naming.
The input is assumed to be a tuple:
    (content_img, style_img)
Both of the should have shape:
    [batch, channel, height, width]
The output will be a tuple shown as below:
    (output_image, output_image_feature_map, style_fature_map)
The output_image should have shape:
    [batch, channel, height, width]
The output_image_feature_map should have shape:
    PENDING
The style_feature_map should have shape:
    PENDING
Both feature maps are used to compute style loss and they should
have the same shape.
The model will automatically apply paddings.
'''
class UAdaINModel(nn.Module):

    def __init__(self, input_channel, has_bn=False):
        super(UAdaINModel, self).__init__()

        #padding layer
        self.pad = nn.ReflectionPad2d(60)

        #in order to make it easier to get feature map
        #contracting path won't be packed in a Sequential model
        self.down1 = Down(input_channel = input_channel,\
                          output_channel = 64,\
                          has_bn = has_bn)
        self.down2 = Down(input_channel = 64,\
                          output_channel = 128,\
                          has_bn = has_bn)
        self.down3 = Down(input_channel = 128,\
                          output_channel = 256,\
                          has_bn = has_bn)
        self.down4 = Down(input_channel = 256,\
                          output_channel = 512,\
                          has_bn = has_bn)
        
        #AdaIN layer
        self.adain = AdaIN(input_channel = 512,\
                           intermediate_channel = 1024,\
                           has_bn = has_bn)

        #in order to get skip connections
        #expanding path won't be packed in a Sequential model
        self.up1 = Up(input_channel = 1024,\
                      intermediate_channel = 512,\
                      output_channel = 256,\
                      has_bn = has_bn) #input should be the output of
                                       #previous layer + skip connection
                                       #so the amount of channels should
                                       #be doubled
        self.up2 = Up(input_channel = 512,\
                      intermediate_channel = 256,\
                      output_channel = 128,\
                      has_bn = has_bn) #as stated above, the input_channel
                                       #should be twice as the output_channel
                                       #of previous layer
        self.up3 = Up(input_channel = 256,\
                      intermediate_channel = 128,\
                      output_channel = 64,\
                      has_bn = has_bn) #as stated above, the input_channel
                                       #should be twice as the output_channel
                                       #of previous layer
        self.up4 = Up(input_channel = 128,\
                      intermediate_channel = 64,\
                      output_channel = 64,\
                      has_bn = has_bn,\
                      last_layer = True) #as stated above, the input_channel
                                         #should be twice as the output_channel
                                         #of previous layer
                                         #as the last up-sampling layer,
                                         #the up-conv should be suppressed

        #final 1x1 conv to change the amount of channels
        self.output = nn.Conv2d(in_channels = 64,\
                                out_channels = 3,\
                                kernel_size = (1,1))

