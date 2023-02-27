from Model.Layer import Down, BottleNeck, Up
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
import os

'''
A classic auto-encoder based on u-net. It's designed
for pretraining process only.
The input is assumed to have shape
    [batch, channel, height, width]
The output should be a tuple shown by following:
    (output_img, input_img_map)
The output_img should have shape:
    [batch, channel, height, width]
Both input_img_map should have shape:
    [batch, 512, height/8, width/8]
The model will automatically apply paddings.
'''
class UNetModel(nn.Module):

    def __init__(self, input_channel, has_bn=False):
        super(UNetModel, self).__init__()

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
        self.bottleneck = BottleNeck(input_channel = 512,\
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
                                      out_channels = 3,\
                                      kernel_size = (1,1)),
                            nn.ReLU()
        )

    def skip_connections(self, down_feature, up_feature):
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

        #crop down_feature
        sc = down_feature[:,:,\
                          height_start_index : height_start_index+up_height,\
                          width_start_index : width_start_index+up_width]
        
        #concat feature
        re = torch.concat((sc,up_feature), dim=1)

        return re

    def forward(self, x):
        #copy and paste from UAdaIN
        content = x

        #apply paddings
        content = self.pad(content)

        #extract feature map of content
        content_feature_map, sc1 = self.down1(content)
        content_feature_map, sc2 = self.down2(content_feature_map)
        content_feature_map, sc3 = self.down3(content_feature_map)
        content_feature_map, sc4 = self.down4(content_feature_map)

        #bottleneck
        latent = self.bottleneck(content_feature_map)

        #upsampling with skip-connections
        latent = self.skip_connections(down_feature = sc4,\
                                       up_feature = latent)
        
        latent = self.up1(latent)
        latent = self.skip_connections(down_feature = sc3,\
                                       up_feature = latent)
        latent = self.up2(latent)
        latent = self.skip_connections(down_feature = sc2,\
                                       up_feature = latent)
        latent = self.up3(latent)
        latent = self.skip_connections(down_feature = sc1,\
                                       up_feature = latent)
        latent = self.up4(latent)

        #output
        out = self.output(latent)

        return out

'''
Application class for U-net,
'''
class UNet:
    
    def __init__(self, input_channel, device = None,\
                 dataset=None,batch_size = 4, lr = 0.001,\
                 has_bn=False, prev_weights=None):
        #initialize device if not specified
        self.device = device
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available()\
                                       else "cpu")

        #initialize model
        self.model = UNetModel(input_channel = input_channel,\
                               has_bn = has_bn)
        if(prev_weights is not None):
            weights = torch.load(prev_weights)
            self.model.load_state_dict(weights)
        self.model.to(self.device)

        #initialize loss func and optimizer
        self.loss = nn.MSELoss()
        self.loss.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        #initialize dataset, loader will be initilized at the
        #beginning at each epoch to ensure shuffling
        self.dataset = dataset
        self.batch_size = batch_size

    def prepad(self, input_):
        #before feeding the model, prepad the input so that it can
        #fulfil the requirements of the model
        
        height_pad = 256 - input_.shape[2] % 256
        width_pad = 256 - input_.shape[3] % 256
        up = height_pad // 2
        down = height_pad - up
        left = width_pad // 2
        right = width_pad - left
        
        padding = nn.ReflectionPad2d((left, right, up, down))

        return padding(input_), up, left

    def save_model(self, model_dir, epoch, step, loss):
        #save weights
        model_name = 'unet_epoch{current_epoch}_step{current_step}_loss{current_loss}.weights'\
                                                    .format(current_epoch = epoch,\
                                                            current_step = step,\
                                                            current_loss = loss)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))

    def train(self, epoch=10, model_dir=None,\
              steps_to_save = 10, maximum_model=5):
        
        #initialize model_dir if not specified
        if not model_dir:
            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%Y-%m-%d')
            model_dir = 'SavedModel/UNet_Model_{date}'.format(date=current_time)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        for i in range(epoch):
            #initialize step and dataloader
            step = 0
            loader = DataLoader(dataset = self.dataset,\
                                batch_size = self.batch_size,\
                                shuffle = True)
            #training process
            for _, data in enumerate(loader):
                #apply pre-pad
                height_ori = data.shape[2]
                width_ori = data.shape[3]
                padded_data, height_start, width_start = self.prepad(data)

                #train one step
                self.optimizer.zero_grad()
                output = self.model(padded_data)
                #only crop the original size
                output = output[:,:,\
                                height_start:height_start+height_ori,\
                                width_start:width_start+width_ori]
                loss_ = self.loss(output, data)
                loss_.backward()
                self.optimizer.step()

                #check save model
                if not step%steps_to_save:
                    self.save_model(model_dir, i, step, loss_)

                    #check model num
                    file_list = os.listdir(model_dir)
                    if len(file_list) > maximum_model:
                        #remove previous epoch
                        file_list.sort(key=lambda x: os.path.getmtime(
                                                     os.path.join(model_dir, x)))
                        os.remove(os.path.join(model_dir, file_list[0]))

                #next step
                print("Epoch: {e}, Step: {s}, Loss:{l}".format(e = i,\
                                                               s = step,\
                                                               l = loss_))
                step = step + 1

            #save the model after epoch
            self.save_model(model_dir, i, step, loss_)

            #check model num
            file_list = os.listdir(model_dir)
            if len(file_list) > maximum_model:
                #remove previous epoch
                file_list.sort(key=lambda x: os.path.getmtime(
                                             os.path.join(model_dir, x)))
                os.remove(os.path.join(model_dir, file_list[0]))

    def pred(self, input_):
        #make sure it contains batch
        if len(input_.shape) != 4:
            input_ = torch.unsqueeze(input_, dim=0)
        
        #calculate result
        with torch.no_grad():
            output = self.model(input_)
        
        #remove empty axis
        output = output.squeeze()

        #make sure it's on cpu
        output = output.cpu()

        #permute [C, H, W] -> [H, W, C]
        output = output.permute((1,2,0))

        #clamp result -> [0,1] (0 is actually guaranteed by relu)
        output = torch.clamp(output, min=0.0, max=1.0)
        
        return output.numpy()