from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            #self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            #if self.previous_conv:
            #    self.bn = nn.BatchNorm2d(int(input_channels/size), eps=1e-4, momentum=0.1, affine=True)
            #else:
            #    self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        #self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        #x = self.bn(x)
        x = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        #x = self.relu(x)
        return x

class MLP_S(nn.Module):
    def __init__(self):
        super(MLP_S, self).__init__()   
		
        self.conv1 = nn.Conv2d(1, 784, kernel_size=28, stride=1)
        #self.conv1 = nn.Linear(784, 784)
        #self.conv1 = BinConv2d(1, 784, kernel_size=28, stride=1, padding=0)
        #self.bin_FC2_Golden = BinConv2d(784, 500, Linear=True, previous_conv=True, size=1*1)
        self.bin_FC2_1 = BinConv2d(392, 500, Linear=True, previous_conv=True, size=1*1)
        self.bin_FC2_2 = BinConv2d(392, 500, Linear=True, previous_conv=True, size=1*1)
        #self.bin_FC2 = BinConv2d(784, 500, Linear=True, previous_conv=True, size=1*1)
        self.bin_FC3 = BinConv2d(500, 250, Linear=True, previous_conv=False, size=4*4)
        self.FC4 = nn.Linear(250, 10)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        
        #print(x.size())
        x = self.conv1(x)
        #print(x.size())
        #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        #print(x)
		
        #p_golden = self.bin_FC2_Golden(x)
        p1 = self.bin_FC2_1(x[:,0:392,:,:])
        p2 = self.bin_FC2_2(x[:,392:784,:,:])
        #x = self.bin_FC2(x)
        #print(p1)
        x = p1 + p2
        #print(np.array_equal(x,p_golden))
        #print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
        #print(x)
        x = self.bin_FC3(x)
        x = self.FC4(x)
        return x
