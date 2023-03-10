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
			self.bn = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
			self.conv = nn.Conv2d(input_channels, output_channels,
					kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
		else:
			self.bn = nn.BatchNorm1d(output_channels, eps=1e-4, momentum=0.1, affine=True)
			self.linear = nn.Linear(input_channels, output_channels)
		self.relu = nn.ReLU(inplace=True)
    
	def forward(self, x):
		#x = self.bn(x)
		#print(x.size())
		x = BinActive.apply(x)
		if self.dropout_ratio!=0:
			x = self.dropout(x)
		if not self.Linear:
			x = self.conv(x)
		else:
			x = self.linear(x)
		#x = self.relu(x)
		#print(x.size())
		x = self.bn(x)
		
		return x

class AlexNet_BCIM(nn.Module):

	def __init__(self, num_classes=10):
		super(AlexNet_BCIM, self).__init__()
		self.num_classes = num_classes
		
		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		self.relu1 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    #############################################   
   
		#self.bin_conv2 = BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.bin_conv2_1 = BinConv2d(20, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.bin_conv2_2 = BinConv2d(20, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.bin_conv2_3 = BinConv2d(20, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.bin_conv2_4 = BinConv2d(20, 256, kernel_size=5, stride=1, padding=2, groups=1)
		self.bin_conv2_5 = BinConv2d(16, 256, kernel_size=5, stride=1, padding=2, groups=1)
		
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		
		##############
		#self.bin_conv3 = BinConv2d(256, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv3_1 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv3_2 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv3_3 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv3_4 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1)
		self.bin_conv3_5 = BinConv2d(32, 384, kernel_size=3, stride=1, padding=1)
		
		
		#############
		#self.bin_conv4 = BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_1 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_2 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_3 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_4 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_5 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_6 = BinConv2d(56, 384, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv4_7 = BinConv2d(48, 384, kernel_size=3, stride=1, padding=1, groups=1)
		
		
		##############
		#self.bin_conv5 = BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_1 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_2 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_3 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_4 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_5 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_6 = BinConv2d(56, 256, kernel_size=3, stride=1, padding=1, groups=1)
		self.bin_conv5_7 = BinConv2d(48, 256, kernel_size=3, stride=1, padding=1, groups=1)
				
		self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		
		###############
		#self.bin_conv6 = BinConv2d(256 * 6 * 6, 4096, Linear=True)
		self.bin_conv6_1 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_2 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_3 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_4 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_5 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_6 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_7 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_8 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_9 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_10 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_11 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_12 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_13 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_14 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_15 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_16 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_17 = BinConv2d(512, 4096, Linear=True)
		self.bin_conv6_18 = BinConv2d(512, 4096, Linear=True)
		
		
		#self.bin_conv7 = BinConv2d(4096, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_1 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_2 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_3 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_4 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_5 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_6 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_7 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
		self.bin_conv7_8 = BinConv2d(512, 4096, dropout=0.5, Linear=True)
   
    #############################################
		self.bn2 = nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True)
		self.dropout = nn.Dropout()
		self.linear = nn.Linear(4096, num_classes)
		
        

	def forward(self, x):
        		
		print('input: ', x.size())
		x = self.conv1(x)
		#print('conv1: ', x.size())
		x = self.bn1(x)
		#print('bn1: ',x.size())
		x = self.relu1(x)
		#print('relu1: ',x.size())
		x = self.pool1(x)
		
		print('bin_conv2: ', x.size())
		#x = self.bin_conv2(x)
		p1 = self.bin_conv2_1(x[:,0:20,:,:])
		p2 = self.bin_conv2_2(x[:,20:40,:,:])
		p3 = self.bin_conv2_3(x[:,40:60,:,:])
		p4 = self.bin_conv2_4(x[:,60:80,:,:])
		p5 = self.bin_conv2_5(x[:,80:96,:,:])
		x = p1 + p2 + p3 + p4 + p5
		
		x = self.pool2(x)
		#print('pool2: ',x.size())
			
			
		#x = self.bin_conv3(x)
		print('conv3: ',x.size())
		p1 = self.bin_conv3_1(x[:,0:56,:,:])
		p2 = self.bin_conv3_2(x[:,56:112,:,:])
		p3 = self.bin_conv3_3(x[:,112:168,:,:])
		p4 = self.bin_conv3_4(x[:,168:224,:,:])
		p5 = self.bin_conv3_5(x[:,224:257,:,:])
		x = p1 + p2 + p3 + p4 + p5
				
				
		#x = self.bin_conv4(x)
		print('conv4: ',x.size())
		p1 = self.bin_conv4_1(x[:,0:56,:,:])
		p2 = self.bin_conv4_2(x[:,56:112,:,:])
		p3 = self.bin_conv4_3(x[:,112:168,:,:])
		p4 = self.bin_conv4_4(x[:,168:224,:,:])
		p5 = self.bin_conv4_5(x[:,224:280,:,:])
		p6 = self.bin_conv4_6(x[:,280:336,:,:])
		p7 = self.bin_conv4_7(x[:,336:384,:,:])
		x = p1 + p2 + p3 + p4 + p5 + p6 + p7
		
		#x = self.bin_conv5(x)
		print('conv5: ',x.size())
		p1 = self.bin_conv5_1(x[:,0:56,:,:])
		p2 = self.bin_conv5_2(x[:,56:112,:,:])
		p3 = self.bin_conv5_3(x[:,112:168,:,:])
		p4 = self.bin_conv5_4(x[:,168:224,:,:])
		p5 = self.bin_conv5_5(x[:,224:280,:,:])
		p6 = self.bin_conv5_6(x[:,280:336,:,:])
		p7 = self.bin_conv5_7(x[:,336:384,:,:])
		x = p1 + p2 + p3 + p4 + p5 + p6 + p7
		
		
		x = self.pool3(x)
		#print('pool3: ',x.size())	
		x = self.avgpool(x)
		#print('avgpool: ',x.size())
		x = x.view(x.size(0), 256 * 6 * 6)
		#print('view: ',x.size())
        #x = self.classifier(x)
		
		print('conv6: ',x.size())
		#x = self.bin_conv6(x)
		p1 = self.bin_conv6_1(x[:,0:512])
		p2 = self.bin_conv6_2(x[:,512:1024])
		p3 = self.bin_conv6_3(x[:,1024:1536])
		p4 = self.bin_conv6_4(x[:,1536:2048])
		p5 = self.bin_conv6_5(x[:,2048:2560])
		p6 = self.bin_conv6_6(x[:,2560:3072])
		p7 = self.bin_conv6_7(x[:,3072:3584])
		p8 = self.bin_conv6_8(x[:,3584:4096])
		p9 = self.bin_conv6_9(x[:,4096:4608])
		p10 = self.bin_conv6_10(x[:,4608:5120])
		p11 = self.bin_conv6_11(x[:,5120:5632])
		p12 = self.bin_conv6_12(x[:,5632:6144])
		p13 = self.bin_conv6_13(x[:,6144:6656])
		p14 = self.bin_conv6_14(x[:,6656:7168])
		p15 = self.bin_conv6_15(x[:,7168:7680])
		p16 = self.bin_conv6_16(x[:,7680:8192])
		p17 = self.bin_conv6_17(x[:,8192:8704])
		p18 = self.bin_conv6_18(x[:,8704:9216])
		x = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18
		
		
		print('conv7: ',x.size())
		#x = self.bin_conv7(x)
		p1 = self.bin_conv7_1(x[:,0:512])
		p2 = self.bin_conv7_2(x[:,512:1024])
		p3 = self.bin_conv7_3(x[:,1024:1536])
		p4 = self.bin_conv7_4(x[:,1536:2048])
		p5 = self.bin_conv7_5(x[:,2048:2560])
		p6 = self.bin_conv7_6(x[:,2560:3072])
		p7 = self.bin_conv7_7(x[:,3072:3584])
		p8 = self.bin_conv7_8(x[:,3584:4096])
		x = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
		
		x = self.bn2(x)
		x = self.dropout(x)
		print('linear: ',x.size())
		x = self.linear(x)
		pdb.set_trace()
		
		return x

