import torch
import torch.utils.data as data
import os
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
This class is created to represent a set of images.
It does not support receiving a transform object to transform
images, so every image has to be processed before putting them
into the dataset.
The model will be trained on 256*256 pictures. However, as FCN,
as long as the size of picture can satisfy the pooling layers in
the network, any size is acceptable. 
'''
class ImageDataSet(data.Dataset):

    def __init__(self, source_folder, device):
        #source_folder should be the path to the folder containing
        #images, sub-directories will also be included
        #device should be the device for tensors, every tensor returned
        #by this dataset will be allocated to that device
        super(ImageDataSet, self).__init__()

        self.source_folder = source_folder
        self.images = [file for file in file_generator(source_folder)]
        self.transer = transforms.ToTensor()
        self.device = device

    def __getitem__(self, index):
        #return the item at given index
        #return value should have shape:
        #   [channel, height, width]
        #and the ordering of channels is R G B
        img = Image.open(os.path.join(self.source_folder,\
                                      self.images[index]))
        img_tensor = self.transer(img)
        
        return img_tensor.to(self.device)

    def __len__(self):
        #the length of this dataset
        return len(self.images)
        

