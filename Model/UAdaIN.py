from Model.Layer import Down, AdaIN, Up
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The model. I name it UAdaIN, though I'm bad at naming.
The input is assumed to be a tuple:
    (content_img, style_img)
Both of them should have shape:
    [batch, channel, height, width]
The output will be a tuple shown as below:
    (output_image, transformed_feature map, style_means, style_stds)
The output_image should have shape:
    [batch, channel, height, width]
The input_img_feature_map should have shape:
    [batch, 512, height/8, width/8]
The style_means should be a list containing:
    [style_mean1, style_mean2, style_mean3, style_mean4]
The style_stds should be a list containing:
    [style_std1, style_std2, style_std3, style_std4]
Where each of them (mean and std) should have shape:
    [batch, channel]
The model will automatically apply paddings.
'''
class UAdaINModel(nn.Module):

    def __init__(self, input_channel, has_bn=False):
        super(UAdaINModel, self).__init__()

        #padding layer
        self.pad = nn.ReflectionPad2d(30)

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
        #Using the same name as the one in UNet to load pretrain weights
        self.bottleneck = AdaIN(input_channel = 512,\
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
        self.output = nn.Sequential(
                            nn.Conv2d(in_channels = 64,\
                                      out_channels = input_channel,\
                                      kernel_size = (1,1)),
                            nn.ReLU()
        )

    def freeze_encoder(self):
        #freeze the encoder part
        self.down1.requires_grad_(requires_grad=False)
        self.down2.requires_grad_(requires_grad=False)
        self.down3.requires_grad_(requires_grad=False)
        self.down4.requires_grad_(requires_grad=False)

    def calc_mean_and_std(self, x):
        #helper method to calculate instance mean and standard deviation
        #assuming x.shape = [batch, channel, height, width]
        #output should be a tuple: (mean, std)
        #both should have shape [batch, channel]
        return torch.mean(x, dim=[2,3], keepdim=True),\
               torch.std(x, unbiased = False, dim=[2,3], keepdim=True)+1e-6

    def skip_connections(self, down_feature, up_feature,\
                         style_mean, style_std):
        #helper method for generating skip connections
        #the input of down_feature should be from contracting path with shape:
        #    [batch, channel, down_height, down_width]
        #the input of up_feature should be from expanding path with shape:
        #    [batch, channel, up_height, up_width]
        #though it won't be checked, the shape of input should also satisfy:
        #    up_height <= down_height
        #    up_width <= down_width
        #    2|down_height-up_height
        #    2|down_width-up_width
        #the output should have shape:
        #    [batch, channel*2, up_height, up_width]

        down_height = down_feature.shape[2]
        down_width = down_feature.shape[3]
        up_height = up_feature.shape[2]
        up_width = up_feature.shape[3]

        height_start_index = (down_height-up_height)//2
        width_start_index = (down_width-up_width)//2

        #calc original mean and std
        down_mean, down_std = self.calc_mean_and_std(down_feature)

        #crop down_feature
        sc = down_feature[:,:,\
                          height_start_index : height_start_index+up_height,\
                          width_start_index : width_start_index+up_width]
        print(sc.shape)

        #adain
        sc = style_std * (sc-down_mean)/down_std + style_mean
        print(sc.shape)
        
        #concat feature
        re = torch.concat((sc,up_feature), dim=1)

        return re

    def get_encoded_feature(self, x):
        #the input x will go through contracting part to get
        #encoded feature of x
        #this method is used for computing content loss

        x = self.pad(x)

        map,_ = self.down1(x)
        map_mean1, map_std1 = self.calc_mean_and_std(map)

        map,_ = self.down2(map)
        map_mean2, map_std2 = self.calc_mean_and_std(map)

        map,_ = self.down3(map)
        map_mean3, map_std3 = self.calc_mean_and_std(map)

        map,_ = self.down4(map)
        map_mean4, map_std4 = self.calc_mean_and_std(map)

        return [map_mean1, map_mean2, map_mean3, map_mean4],\
               [map_std1, map_std2, map_std3, map_std4]

    def forward(self, x):
        #extract content and style
        content, style = x

        #apply paddings
        content = self.pad(content)
        style = self.pad(style)

        #extract feature map of content
        content_feature_map, sc1 = self.down1(content)
        content_feature_map, sc2 = self.down2(content_feature_map)
        content_feature_map, sc3 = self.down3(content_feature_map)
        content_feature_map, sc4 = self.down4(content_feature_map)

        #extract feature map of content
        style_feature_map,_ = self.down1(style)
        style_mean1, style_std1 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down2(style_feature_map)
        style_mean2, style_std2 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down3(style_feature_map)
        style_mean3, style_std3 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down4(style_feature_map)
        style_mean4, style_std4 = self.calc_mean_and_std(style_feature_map)

        #adain
        latent, transformed_map = self.bottleneck((content_feature_map, style_feature_map))

        #upsampling with skip-connections
        latent = self.skip_connections(down_feature = sc4,\
                                       up_feature = latent,\
                                       style_mean = style_mean4,\
                                       style_std = style_std4)
        latent = self.up1(latent)
        latent = self.skip_connections(down_feature = sc3,\
                                       up_feature = latent,\
                                       style_mean = style_mean3,\
                                       style_std = style_std3)
        latent = self.up2(latent)
        latent = self.skip_connections(down_feature = sc2,\
                                       up_feature = latent,\
                                       style_mean = style_mean2,\
                                       style_std = style_std2)
        latent = self.up3(latent)
        latent = self.skip_connections(down_feature = sc1,\
                                       up_feature = latent,\
                                       style_mean = style_mean1,\
                                       style_std = style_std1)
        latent = self.up4(latent)

        #output
        out = self.output(latent)

        return out, transformed_map,\
               [style_mean1, style_mean2, style_mean3, style_mean4],\
               [style_std1, style_std2, style_std3, style_std4]
