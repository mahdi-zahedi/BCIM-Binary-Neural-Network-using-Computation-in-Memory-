import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

#__all__ = ['AlexNet', 'alexnet']

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
			Linear=False):
		super(BinConv2d, self).__init__()
		self.layer_type = 'BinConv2d'
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dropout_ratio = dropout
	
		if dropout!=0:
			self.dropout = nn.Dropout(dropout)
		self.Linear = Linear
		if not self.Linear:
			self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
			self.conv = nn.Conv2d(input_channels, output_channels,
					kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
		else:
			self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
			self.linear = nn.Linear(input_channels, output_channels)
		#self.relu = nn.ReLU(inplace=True)
    
	def forward(self, x):
		x = self.bn(x)
		x = BinActive.apply(x)
		if self.dropout_ratio!=0:
			x = self.dropout(x)
		if not self.Linear:
			x = self.conv(x)
		else:
			x = self.linear(x)
		#x = self.relu(x)
		return x

class AlexNet(nn.Module):

	def __init__(self, num_classes=10):
		super(AlexNet, self).__init__()
		self.num_classes = num_classes
		
		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		self.relu1 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.bin_conv2 = BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.bin_conv3 = BinConv2d(256, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv4 = BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5 = BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
				
        #self.features = nn.Sequential(
        #    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        #    nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #    BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #    BinConv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #    BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1),
        #    BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #)
		
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.bin_conv6 = BinConv2d(256 * 6 * 6, 4096, Linear=True)
		self.bin_conv7 = BinConv2d(4096, 4096, dropout=0.5, Linear=True)
		self.bn2 = nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True)
		self.dropout = nn.Dropout()
		self.linear = nn.Linear(4096, num_classes)
		
        #self.classifier = nn.Sequential(
        #    BinConv2d(256 * 6 * 6, 4096, Linear=True),
        #    BinConv2d(4096, 4096, dropout=0.5, Linear=True),
        #    nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
        #   nn.Dropout(),
        #   nn.Linear(4096, num_classes),
        #)

	def forward(self, x):
        #x = self.features(x)
		
		#print('input: ', x.size())
		x = self.conv1(x)
		#print('conv1: ', x.size())
		x = self.bn1(x)
		#print('bn1: ',x.size())
		x = self.relu1(x)
		#print('relu1: ',x.size())
		x = self.pool1(x)
		#print('pool1: ',x.size())
		x = self.bin_conv2(x)
		#print('conv2: ',x.size())
		x = self.pool2(x)
		#print('pool2: ',x.size())
		x = self.bin_conv3(x)
		#print('conv3: ',x.size())
		x = self.bin_conv4(x)
		#print('conv4: ',x.size())
		x = self.bin_conv5(x)
		#print('conv5: ',x.size())
		x = self.pool3(x)
		#print('pool3: ',x.size())	
		
		x = self.avgpool(x)
		#print('avgpool: ',x.size())
		x = x.view(x.size(0), 256 * 6 * 6)
		#print('view: ',x.size())
        #x = self.classifier(x)
		
		x = self.bin_conv6(x)
		x = self.bin_conv7(x)
		x = self.bn2(x)
		x = self.dropout(x)
		x = self.linear(x)
		
		return x


#def alexnet(pretrained=False, **kwargs):
#    r"""AlexNet model architecture from the
#    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = AlexNet(**kwargs)
#    if pretrained:
#        model_path = 'model_list/alexnet.pth.tar'
#        pretrained_model = torch.load(model_path)
#        model.load_state_dict(pretrained_model['state_dict'])
#    return model
