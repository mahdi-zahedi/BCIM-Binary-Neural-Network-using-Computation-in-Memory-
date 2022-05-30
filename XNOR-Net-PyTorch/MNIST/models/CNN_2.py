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

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()   
		
        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		
        self.bin_FC2_1 = BinConv2d(512, 1210, Linear=True, previous_conv=True, size=11*11)
        self.bin_FC2_2 = BinConv2d(512, 1210, Linear=True, previous_conv=True, size=11*11)
        self.bin_FC2_3 = BinConv2d(186, 1210, Linear=True, previous_conv=True, size=11*11)
        
        self.bin_FC3_1 = BinConv2d(512, 1210, Linear=True, previous_conv=False, size=1*1)
        self.bin_FC3_2 = BinConv2d(512, 1210, Linear=True, previous_conv=False, size=1*1)
        self.bin_FC3_3 = BinConv2d(186, 1210, Linear=True, previous_conv=False, size=1*1)	
		
        self.FC4 = nn.Linear(1210, 10)

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
        
        ##########################################
        x = self.conv1(x)
        x = self.pool1(x)
        ##########################################
        
        x= torch.reshape(x, (x.size(0), 1210))
        #print(x.size())
        p1 = self.bin_FC2_1(x[:,0:512])
        p2 = self.bin_FC2_2(x[:,512:1024])
        p3 = self.bin_FC2_3(x[:,1024:1210])
        #p1=p1.sign()
        #p2=p2.sign()
        
        x = p1+p2+p3
        
        ##########################################
		
        p1 = self.bin_FC3_1(x[:,0:512])
        p2 = self.bin_FC3_2(x[:,512:1024])
        p3 = self.bin_FC3_3(x[:,1024:1210])
        #p1=p1.sign()
        #p2=p2.sign()
		
        x = p1 + p2 + p3      
        
		############################################
        x = self.FC4(x)
        return x
