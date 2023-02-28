import torch
import torch.utils.data as data
import os
import random
from PIL import Image
from torchvision import transforms

'''
A generator method to return only files
'''
def file_generator(path):
    for dir, _, files in os.walk(path):
        for file in files:
            yield os.path.join(dir, file)

'''
This class is created to represent a set of content and styles.
It does not support receiving a transform object to transform
images, so every image has to be processed before putting them
into the dataset.
The model will be trained on 256*256 pictures. However, as FCN,
as long as the size of picture can satisfy the pooling layers in
the network, any size is acceptable. 
Also, this dataset is only focusing on content. That is, one epoch
means every content has been used once. It does not generate fixed
content-style pairs. Instead, for each content it returned, it will
randomly pick one style image.
'''
class ContentStyleDataSet(data.Dataset):

    def __init__(self, content_folder, style_folder, device):
        #source_folder should be the path to the folder containing
        #images, sub-directories will also be included
        #device should be the device for tensors, every tensor returned
        #by this dataset will be allocated to that device
        super(ContentStyleDataSet, self).__init__()

        self.content_folder = content_folder
        self.content_img = [file for file in file_generator(content_folder)]
        
        self.style_folder = style_folder
        self.style_img = [file for file in file_generator(style_folder)]

        self.transer = transforms.ToTensor()
        self.device = device

    def __getitem__(self, index):
        #return the item at given index
        #return value should be a tuple shown below:
        #   (content, style)
        #both of them should have shape:
        #   [channel, height, width]
        #and the ordering of channels is R G B
        content = Image.open(self.content_img[index])
        content_tensor = self.transer(content)

        style = Image.open(random.choice(self.style_img))
        style_tensor = self.transer(style)
        
        return content_tensor.to(self.device),\
               style_tensor.to(self.device)

    def __len__(self):
        #the length of this dataset
        return len(self.content_img)