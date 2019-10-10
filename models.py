## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.ReLU()
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #maxpool layer
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        #Second conv layer: 32 input 
        self.conv2 = nn.Conv2d(32,64,5)
        self.relu2 = nn.ReLU()
        
        #maxpool layer
        self.maxpool2 = nn.MaxPool2d(2, 2)
                    
        #Fully connected 1
        self.fc1 = nn.Linear(64*4*4, 136)
        
        # dropout with p=0.4
        #self.fc1_drop = nn.Dropout(p=0.4)
        
         # finally, create 136 output channels 
        #self.fc2 = nn.Linear(50, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
         # Convolution 1
        x = self.conv1(x)
        x = self.relu1(x)
        # Max pool 1
        x = self.maxpool1(x)

        # Convolution 2 
        x = self.conv2(x)
        x = self.relu2(x)

        # Max pool 2 
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        
        #Linear function 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        return x  
        
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        
        #x = x.view(x.size(0), -1)
         
        #x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        #x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
         
        #return x

