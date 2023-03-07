from Model.Layer import Down, AdaIN, Up
import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import datetime, os

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

    def __init__(self, input_channel, has_bn=False, sc_adain=True):
        super(UAdaINModel, self).__init__()
        self.has_bn = has_bn
        self.sc_adain = sc_adain

        #padding layer
        self.pad = nn.ReflectionPad2d(62)

        #in order to make it easier to get feature map
        #contracting path won't be packed in a Sequential model
        #this is the same as vgg-13
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
        self.down5 = Down(input_channel = 512,\
                          output_channel = 512,\
                          has_bn = has_bn)
        
        #AdaIN layer
        #Using the same name as the one in UNet to load pretrain weights
        self.bottleneck = AdaIN(input_channel = 512,\
                           intermediate_channel = 1024,\
                           has_bn = has_bn)

        #in order to get skip connections
        #expanding path won't be packed in a Sequential model
        #this is a mirror of vgg-13
        self.up1 = Up(input_channel = 1024,\
                      intermediate_channel = 512,\
                      output_channel = 512,\
                      has_bn = has_bn) #input should be the output of
                                       #previous layer + skip connection
                                       #so the amount of channels should
                                       #be doubled
        self.up2 = Up(input_channel = 1024,\
                      intermediate_channel = 512,\
                      output_channel = 256,\
                      has_bn = has_bn) #as stated above, the input_channel
                                       #should be twice as the output_channel
                                       #of previous layer
        self.up3 = Up(input_channel = 512,\
                      intermediate_channel = 256,\
                      output_channel = 128,\
                      has_bn = has_bn) #as stated above, the input_channel
                                       #should be twice as the output_channel
                                       #of previous layer
        self.up4 = Up(input_channel = 256,\
                      intermediate_channel = 128,\
                      output_channel = 64,\
                      has_bn = has_bn) #as stated above, the input_channel
                                       #should be twice as the output_channel
                                       #of previous layer
        self.up5 = Up(input_channel = 128,\
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
        self.down5.requires_grad_(requires_grad=False)

    def load_vgg_weights(self):
        #load vgg weights to contracting path
        vgg = torchvision.models.vgg13_bn(weights='DEFAULT') if self.has_bn\
            else torchvision.models.vgg13(weights='DEFAULT')

        #copy and paste weights
        for uadain_param, vgg_param in zip(self.named_parameters(),\
                                       vgg.features.named_parameters()):
            with torch.no_grad():
                uadain_param[1].requires_grad_(requires_grad=False)
                uadain_param[1].copy_(vgg_param[1])

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

        #adain
        if self.sc_adain:
            sc = style_std * (sc-down_mean)/down_std + style_mean
        
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

        map,_ = self.down5(map)
        map_mean5, map_std5 = self.calc_mean_and_std(map)

        return map,\
               [map_mean1, map_mean2, map_mean3, map_mean4, map_mean5],\
               [map_std1, map_std2, map_std3, map_std4, map_std5]

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
        content_feature_map, sc5 = self.down5(content_feature_map)

        #extract feature map of content
        style_feature_map,_ = self.down1(style)
        style_mean1, style_std1 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down2(style_feature_map)
        style_mean2, style_std2 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down3(style_feature_map)
        style_mean3, style_std3 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down4(style_feature_map)
        style_mean4, style_std4 = self.calc_mean_and_std(style_feature_map)

        style_feature_map,_ = self.down5(style_feature_map)
        style_mean5, style_std5 = self.calc_mean_and_std(style_feature_map)

        #adain
        latent, transformed_map = self.bottleneck((content_feature_map, style_feature_map))

        #upsampling with skip-connections
        latent = self.skip_connections(down_feature = sc5,\
                                       up_feature = latent,\
                                       style_mean = style_mean5,\
                                       style_std = style_std5)
        latent = self.up1(latent)
        latent = self.skip_connections(down_feature = sc4,\
                                       up_feature = latent,\
                                       style_mean = style_mean4,\
                                       style_std = style_std4)
        latent = self.up2(latent)
        latent = self.skip_connections(down_feature = sc3,\
                                       up_feature = latent,\
                                       style_mean = style_mean3,\
                                       style_std = style_std3)
        latent = self.up3(latent)
        latent = self.skip_connections(down_feature = sc2,\
                                       up_feature = latent,\
                                       style_mean = style_mean2,\
                                       style_std = style_std2)
        latent = self.up4(latent)
        latent = self.skip_connections(down_feature = sc1,\
                                       up_feature = latent,\
                                       style_mean = style_mean1,\
                                       style_std = style_std1)
        latent = self.up5(latent)

        #output
        out = self.output(latent)

        return out, transformed_map,\
               [style_mean1, style_mean2, style_mean3, style_mean4, style_mean5],\
               [style_std1, style_std2, style_std3, style_std4, style_std5]

'''
Loss function for UAdaIN
'''
class UAdaINLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super(UAdaINLoss, self).__init__()
        self.mse = F.mse_loss
        self.alpha = alpha
    
    def forward(self, x):
        #expecting x to be a tuple containing:
        #   (out_map, transformed_map, out_means, style_means, out_stds, style_stds)
        out_map, transformed_map, out_means, style_means, out_stds, style_stds = x
        content_loss = self.mse(out_map, transformed_map)

        style_loss = 0
        for i in range(len(out_means)):
            #compute mean loss
            mean_loss = self.mse(out_means[i], style_means[i])
            #compute std means
            std_loss = self.mse(out_stds[i], style_stds[i])
            #addup
            total_loss = mean_loss + std_loss
            style_loss = style_loss + total_loss

        return content_loss + self.alpha * style_loss

'''
Application class for UAdaIN,
'''
class UAdaIN:
    
    def __init__(self, input_channel=3, device = None,\
                 dataset = None,batch_size = 4, lr = 0.001,\
                 has_bn = False, sc_adain = True, prev_weights = None,\
                 alpha = 0.5, pretrain = False):
        #initialize device if not specified
        self.device = device
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available()\
                                       else "cpu")
        self.pretrain = pretrain

        #initialize model
        self.model = UAdaINModel(input_channel = input_channel,\
                               has_bn = has_bn, sc_adain = sc_adain)
        if(prev_weights is not None):
            weights = torch.load(prev_weights)
            self.model.load_state_dict(weights)
        else:
            if input_channel == 3:
                self.model.load_vgg_weights()
            else:
                print('Using default weights is not encouraged,'+\
                      ' please pretrain your own network first.')
        self.model.to(self.device)
        self.model.freeze_encoder()

        #initialize loss func and optimizer
        self.loss = nn.MSELoss() if pretrain\
                    else UAdaINLoss(alpha)
        self.loss.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        #initialize dataset, loader will be initilized at the
        #beginning at each epoch to ensure shuffling
        self.dataset = dataset
        self.batch_size = batch_size

    def prepad(self, input_):
        #before feeding the model, prepad the input so that it can
        #fulfil the requirements of the model
        
        height_pad = 32 - input_.shape[2] % 32
        width_pad = 32 - input_.shape[3] % 32
        up = height_pad // 2
        down = height_pad - up
        left = width_pad // 2
        right = width_pad - left
        
        padding = nn.ReflectionPad2d((left, right, up, down))

        return padding(input_), up, left

    def save_model(self, model_dir, epoch, step, loss, maximum_model):
        #save weights
        model_name = 'uadain_epoch{current_epoch}_step{current_step}_loss{current_loss}.weights'\
                                                    .format(current_epoch = epoch,\
                                                            current_step = step,\
                                                            current_loss = loss)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))

        #check model num
        file_list = os.listdir(model_dir)
        if len(file_list) > maximum_model:
            #remove previous epoch
            file_list.sort(key=lambda x: os.path.getmtime(
                                         os.path.join(model_dir, x)))
            os.remove(os.path.join(model_dir, file_list[0]))

    def train(self, epoch=10, model_dir=None,\
              steps_to_save = 10, maximum_model=5,\
              display = True):
        
        #initialize model_dir if not specified
        if not model_dir:
            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%Y-%m-%d')
            model_dir = 'SavedModel/UAdaIN_Model_{date}'.format(date=current_time)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        for i in range(epoch):
            #initialize step and dataloader
            step = 0
            loader = DataLoader(dataset = self.dataset,\
                                batch_size = self.batch_size,\
                                shuffle = True)
            avg_loss = []
            avg_save_loss = []

            #training process
            for _, data in enumerate(loader):
                content, style = data
                if self.pretrain:
                    style = content

                #apply pre-pad
                height_ori = content.shape[2]
                width_ori = content.shape[3]
                padded_content, height_start, width_start = self.prepad(content)
                padded_style, _, _ = self.prepad(style)

                #train one step
                self.optimizer.zero_grad()
                output = self.model((padded_content, padded_style))

                #unpack output
                out_img, transformed_map, style_means, style_stds = output

                #calculate loss
                if self.pretrain:
                    out_img = out_img[:,:,\
                                      height_start:height_start+height_ori,\
                                      width_start:width_start+width_ori]
                    loss_ = self.loss(out_img, content)
                else:
                    #calculate new featuremap
                    out_map, out_means, out_stds = self.model.get_encoded_feature(out_img)
                    loss_ = self.loss((out_map, transformed_map,\
                                       out_means, style_means,\
                                       out_stds, style_stds))
                loss_.backward()
                self.optimizer.step()

                #save loss
                avg_loss.append(loss_.detach())
                avg_save_loss.append(loss_.detach())

                #check save model
                if not (step%steps_to_save and step):
                    self.save_model(model_dir, i, step,\
                                    sum(avg_save_loss)/len(avg_save_loss),\
                                    maximum_model)
                    avg_save_loss = []

                #clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                #next step
                print("Epoch: {e}, Step: {s}, Loss:{l}".format(e = i,\
                                                               s = step,\
                                                               l = loss_))
                step = step + 1

            #save the model after epoch
            self.save_model(model_dir, i, step,\
                            sum(avg_loss)/len(avg_loss),\
                            maximum_model)
            
            #show the result
            if display:
                out_img = out_img.detach().cpu().squeeze().permute((1,2,0)).numpy()
                plt.imshow(out_img)
                plt.show()

    def pred(self, content, style):
        #make sure it contains batch
        if len(content.shape) != 4:
            content = torch.unsqueeze(content, dim=0)
        if len(style.shape) != 4:
            style = torch.unsqueeze(style, dim=0)

        #prepading
        height_ori = content.shape[2]
        width_ori = content.shape[3]
        padded_content, height_start, width_start = self.prepad(content)
        padded_style, _, _ = self.prepad(style)
        
        #calculate result
        with torch.no_grad():
            output = self.model((padded_content, padded_style))[0]

        #crop the original size
        output = output[:,:,\
                        height_start:height_start+height_ori,\
                        width_start:width_start+width_ori]

        #remove empty axis
        output = output.squeeze()

        #make sure it's on cpu
        output = output.cpu()

        #permute [C, H, W] -> [H, W, C]
        output = output.permute((1,2,0))

        #clamp result -> [0,1] (0 is actually guaranteed by relu)
        output = torch.clamp(output, min=0.0, max=1.0)
        
        return output.numpy()