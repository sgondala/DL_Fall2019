import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


vgg16 = [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'] 

class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        self.conv_model = self.make_model(vgg16)
        print(self.conv_model)
        self.linear_layer = nn.Linear(512, 10)

    def make_model(self, model_list):
        '''
        Arguments:
            model_list (tuple): A tuple with list of layers in the model
        
        Returns:
            A constructed model as per the layers
        '''
        layers = []
        in_channels = 3
        for layer_type in model_list:
            if layer_type == 'P':
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2)) #Fixed for VGG
            else:
                out_channels = layer_type
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
                layers.append(nn.ReLU())
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = self.conv_model(images)
        scores = self.linear_layer(scores.reshape((images.shape[0], -1)))
        # print(scores)
        return scores