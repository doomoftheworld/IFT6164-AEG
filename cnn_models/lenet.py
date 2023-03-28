'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch import rand
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self, nc=3, h=32, w=32):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 6, 5)
        h, w       = round((h-4)/2), round((w-4)/2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        h, w       = round((h-4)/2), round((w-4)/2)
        self.fc1   = nn.Linear(16*h*w, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MadryLeNet(nn.Module):
    def __init__(self, nc=1, h=28, w=28):
        super(MadryLeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.fc1   = nn.Linear(7*7*64, 1024)
        self.fc2   = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, padding=1)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, padding=1)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out   

"""
Self defined structures
"""
# Convolutional layer
class Basic_Conv2d(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d(x)
        return x

# Maxpool Layer
class Basic_Maxpool2d(nn.Module):
    def __init__(self, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.maxpool2d = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_stride, padding=pool_pad),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.maxpool2d(x)
        return x
    
class Basic_Conv2d_with_batch_norm(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d_bn = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(nb_out_channels, eps=0.001),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d_bn(x)
        return x


# Convolutional block
class Conv_Block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.conv = Basic_Conv2d(nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad)
        self.maxpool = Basic_Maxpool2d(pool_k, pool_stride, pool_pad)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x

# Convolutional block with batch normalization
class Conv_Block_with_batch_norm(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.conv = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad)
        self.maxpool = Basic_Maxpool2d(pool_k, pool_stride, pool_pad)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class ConfigCNN(nn.Module):
    def __init__(self, nb_conv, nc=1, h=28, w=28):
        super(ConfigCNN, self).__init__()
        nb_channel = 16
        conv_modules = []
        conv_modules.append(Conv_Block(nc,nb_channel,
                                        *(5,1,2),*(2,2,0)))
        for _ in range(nb_conv-1):
            conv_modules.append(Conv_Block(nb_channel,nb_channel*2,
                                            *(5,1,2),*(2,2,0)))
            nb_channel *= 2
        self.conv_net = nn.Sequential(*conv_modules)
        self.fc1   = nn.Linear(self.get_linear_input_size(nc, h, w), 1024)
        self.fc2   = nn.Linear(1024, 10)

    def get_linear_input_size(self, depth_image, image_h_size, image_w_size):
        rand_input = Variable(rand(1, depth_image, image_h_size, image_w_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size
    
    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out  
    

class ConfigCNN_BatchNorm(nn.Module):
    def __init__(self, nb_conv, nc=1, h=28, w=28):
        super(ConfigCNN_BatchNorm, self).__init__()
        nb_channel = 16
        conv_modules = []
        conv_modules.append(Conv_Block_with_batch_norm(nc,nb_channel,
                                        *(5,1,2),*(2,2,0)))
        for _ in range(nb_conv-1):
            conv_modules.append(Conv_Block_with_batch_norm(nb_channel,nb_channel*2,
                                            *(5,1,2),*(2,2,0)))
            nb_channel *= 2
        self.conv_net = nn.Sequential(*conv_modules)
        self.fc1   = nn.Linear(self.get_linear_input_size(nc, h, w), 1024)
        self.fc2   = nn.Linear(1024, 10)

    def get_linear_input_size(self, depth_image, image_h_size, image_w_size):
        rand_input = Variable(rand(1, depth_image, image_h_size, image_w_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size
    
    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out  