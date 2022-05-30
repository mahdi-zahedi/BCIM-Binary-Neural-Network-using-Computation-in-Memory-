from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=1, stride=1, padding=1, groups=1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        #self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        #self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        #pdb.set_trace()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        #self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        #x = self.bn(x)
        x = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        #pdb.set_trace()
        x = self.conv(x)
        #x = self.relu(x)
        return x

class nin_without_bn(nn.Module):
    def __init__(self):
        super(nin_without_bn, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        #self.relu1 = nn.ReLU(inplace=True)
        self.bin_conv2 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.bin_conv3 = BinConv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bin_conv4 = BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5)
        self.bin_conv5 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bin_conv6 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bin_conv7 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bin_conv8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv9 = nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
		
        return 

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu1(x)
        #print(x.size())
        x = self.bin_conv2(x)
        #print(x.size())
        x = self.bin_conv3(x)
        #print(x.size())
        x = self.pool1(x)
        #print(x.size())
        x = self.bin_conv4(x)
        #print(x.size())
        x = self.bin_conv5(x)
        #print(x.size())
        x = self.bin_conv6(x)
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        x = self.bin_conv7(x)
        #print(x.size())
        x = self.bin_conv8(x)
        x = self.bn2(x)
        #print(x.size())
        x = self.conv9(x)
        #print(x.size())
        x = self.relu2(x)
        
        x = self.pool3(x)
        #print(x.size())
        #pdb.set_trace()
        #x = x.view(x.size(0), 10)
        return x
