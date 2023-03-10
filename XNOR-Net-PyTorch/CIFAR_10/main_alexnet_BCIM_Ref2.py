from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import time
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import models
import util
from torch.autograd import Variable


# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
					
parser.add_argument('--cpu', action='store_true', default = False, help='set if only CPU is available')	
				
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('--caffe-data',  default=False, action='store_true',
                    help='whether use caffe-data')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', action='store',
                    default=None, help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--ref_relative_distance', type=int, default=40, metavar='N',
            help='Relative ref distance to the middel')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')



# define global bin_op
bin_op = None

def main():

	global args
	args = parser.parse_args()
	
	# set the seed
	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	
	###################################################################
    ################### data loading #################################
	##################################################################
	
	
	transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	#transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	#test_transforms = transforms.Compose([transforms.Resize((70, 70)), transforms.CenterCrop((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_size = 4 

	global trainloader, testloader
	
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
	
	
	# define classes
	classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	#######################################################################
	global file1
	global file2
	#file1 = open('acc_%s.txt' % args.arch, 'w')
	#file1 = open('acc_alexnet_no_Relu_All_approximate.txt', 'w')
	file2 = open('extra_info.txt', 'w')
	#######################################################################
	
    # create model
	print('==> building model',args.arch,'...')
	if args.arch=='alexnet':
		model = models.AlexNet()
		input_size = 227
	if args.arch == 'alexnet_BCIM':
		model = models.AlexNet_BCIM()
		input_size = 227
	else:
		raise Exception(args.arch+' is currently not supported')

    #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #    model.features = torch.nn.DataParallel(model.features)
    #    model.cuda()
    #else:
    #    model = torch.nn.DataParallel(model).cuda()


	
	#############################################################################################
	#############################################################################################
	global best_acc
	if not args.pretrained:
		print('==> Initializing model parameters ...')
		best_acc = 0
		for m in model.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				c = float(m.weight.data[0].nelement())
				m.weight.data = m.weight.data.normal_(0, 2.0/c)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data = m.weight.data.zero_().add(1.0)
				m.bias.data = m.bias.data.zero_()
	else:
		print('==> Load pretrained model form', args.pretrained, '...')
		pretrained_model = torch.load(args.pretrained)
		best_acc = pretrained_model['best_acc']
		model.load_state_dict(pretrained_model['state_dict'])
	
	
	if not args.cpu:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=[0])
		
	#print(model)
	
	#############################################################################################
	#############################################################################################
	
	# define loss function (criterion) and optimizer
	# define solver and criterion
	base_lr = float(args.lr)
	param_dict = dict(model.named_parameters())
	params = []
	
	for key, value in param_dict.items():
		params += [{'params':[value], 'lr': base_lr,
			'weight_decay':0.00001}]
	
	global optimizer, criterion
	optimizer = optim.Adam(params, lr=base_lr,weight_decay=0.00001)
	criterion = nn.CrossEntropyLoss().cuda()
	
	#criterion = nn.CrossEntropyLoss().cuda()
	#optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
	##############################################################################################
	

    # define the binarization operator
	global bin_op
	bin_op = util.BinOp(model)
	
	if args.evaluate:
		test(model, args.epochs)
		exit(0)
	
	for epoch in range(1, args.epochs):
		adjust_learning_rate(optimizer, epoch)
	
		# train for one epoch
		train(model, epoch)
		# evaluate on validation set
		test(model, epoch)
	


def train(model, epoch):
		
	# switch to train mode
	model.train()

    #end = time.time()
	for batch_idx, (data, target) in enumerate(trainloader):
	
		data, target = Variable(data.cuda()), Variable(target.cuda())
	
		# process the weights including binarization
		#optimizer.zero_grad()
		bin_op.binarization()
		
		# compute output
		output = model(data)
		loss = criterion(output, target)
	
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
	
		# restore weights
		bin_op.restore()
		bin_op.updateBinaryGradWeight()
	
		optimizer.step()
	
	
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
				epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.data.item(), optimizer.param_groups[0]['lr']))
		
	return


def test(model, epoch):
	
	# switch to evaluate mode
	model.eval()
	test_loss = 0
	correct = 0
	#end = time.time()
	bin_op.binarization()
	iteration = 0
	p1_0_cnt = 0
	p1_P_cnt = 0
	p1_N_cnt = 0	
	p2_0_cnt = 0
	p2_P_cnt = 0
	p2_N_cnt = 0	
	p3_0_cnt = 0
	p3_P_cnt = 0
	p3_N_cnt = 0	
	p4_0_cnt = 0
	p4_P_cnt = 0
	p4_N_cnt = 0
	p5_0_cnt = 0
	p5_P_cnt = 0
	p5_N_cnt = 0
	p6_0_cnt = 0
	p6_P_cnt = 0
	p6_N_cnt = 0	
	p7_0_cnt = 0
	p7_P_cnt = 0
	p7_N_cnt = 0
	p8_0_cnt = 0
	p8_P_cnt = 0
	p8_N_cnt = 0
	
	for data, target in testloader:
		iteration = iteration + 1
		#print(iteration)
		data, target = Variable(data.cuda()), Variable(target.cuda())
	
		# compute output		
		######################################################################################
		######################################################################################
		######################################################################################
		#output = model(data)		
		#print(data.size())
		
		x = model.module.conv1(data)
		x = model.module.bn1(x)
		x = model.module.relu1(x)
		x = model.module.pool1(x)
		##
		p1 = 30*model.module.bin_conv2_1(x[:,0:20,:,:])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
		##		
		p2 = 30*model.module.bin_conv2_2(x[:,20:40,:,:])
		p2_0 = p2.sign() 
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		##		
		p3 = 30*model.module.bin_conv2_3(x[:,40:60,:,:])
		p3_0 = p3.sign()        
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		##		
		p4 = 30*model.module.bin_conv2_4(x[:,60:80,:,:])
		p4_0 = p4.sign()     
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		##		
		p5 = 30*model.module.bin_conv2_5(x[:,80:96,:,:])		
		p5_0 = p5.sign()     
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()		
				
		x = (p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) 
		#x =  p1_0+p2_0+p3_0+p4_0+p5_0
		#x = p1 + p2 + p3 + p4 + p5
		
		
		#for i in range(x.size(0)):
		#	for j in range(x.size(1)):
		#		for k in range(x.size(2)):
		#			for l in range(x.size(3)):
		#				if(p1_0[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p1_0_cnt = p1_0_cnt -1
		#				else:
		#					p1_0_cnt = p1_0_cnt +1
		#				if(p1_P[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p1_P_cnt = p1_P_cnt -1
		#				else:
		#					p1_P_cnt = p1_P_cnt +1
		#				if(p1_N[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p1_N_cnt = p1_N_cnt -1
		#				else:
		#					p1_N_cnt = p1_N_cnt +1
		#					
		#				if(p2_0[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p2_0_cnt = p2_0_cnt -1
		#				else:
		#					p2_0_cnt = p2_0_cnt +1
		#				if(p2_P[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p2_P_cnt = p2_P_cnt -1
		#				else:
		#					p2_P_cnt = p2_P_cnt +1
		#				if(p2_N[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p2_N_cnt = p2_N_cnt -1
		#				else:
		#					p2_N_cnt = p2_N_cnt +1
		#				
		#				
		#				if(p3_0[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p3_0_cnt = p3_0_cnt -1
		#				else:
		#					p3_0_cnt = p3_0_cnt +1
		#				if(p3_P[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p3_P_cnt = p3_P_cnt -1
		#				else:
		#					p3_P_cnt = p3_P_cnt +1
		#				if(p3_N[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p3_N_cnt = p3_N_cnt -1
		#				else:
		#					p3_N_cnt = p3_N_cnt +1
		#					
		#				if(p4_0[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p4_0_cnt = p4_0_cnt -1
		#				else:
		#					p4_0_cnt = p4_0_cnt +1
		#				if(p4_P[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p4_P_cnt = p4_P_cnt -1
		#				else:
		#					p4_P_cnt = p4_P_cnt +1
		#				if(p4_N[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p4_N_cnt = p4_N_cnt -1
		#				else:
		#					p4_N_cnt = p4_N_cnt +1
		#				
		#				if(p5_0[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p5_0_cnt = p5_0_cnt -1
		#				else:
		#					p5_0_cnt = p5_0_cnt +1
		#				if(p5_P[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p5_P_cnt = p5_P_cnt -1
		#				else:
		#					p5_P_cnt = p5_P_cnt +1
		#				if(p5_N[i,j,k,l]!=x[i,j,k,l].sign()):
		#					p5_N_cnt = p5_N_cnt -1
		#				else:
		#					p5_N_cnt = p5_N_cnt +1
					
		#print(p1_0_cnt)
		#print(p1_P_cnt)
		#print(p1_N_cnt)
		#print(p2_0_cnt)
		#print(p2_P_cnt)
		#print(p2_N_cnt)
		#print(p3_0_cnt)
		#print(p3_P_cnt)
		#print(p3_N_cnt)
		#print(p4_0_cnt)
		#print(p4_P_cnt)
		#print(p4_N_cnt)
		#print(p5_0_cnt)
		#print(p5_P_cnt)
		#print(p5_N_cnt)		
		
		#########################################################################
		x = model.module.pool2(x)
		
		p1 = 25*model.module.bin_conv3_1(x[:,0:56,:,:])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
		
		p2 = 25*model.module.bin_conv3_2(x[:,56:112,:,:])
		p2_0 = p2.sign()       
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		
		p3 = 25*model.module.bin_conv3_3(x[:,112:168,:,:])
		p3_0 = p3.sign()       
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		
		p4 = 25*model.module.bin_conv3_4(x[:,168:224,:,:])
		p4_0 = p4.sign()       
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		
		p5 = 25*model.module.bin_conv3_5(x[:,224:257,:,:])
		p5_0 = p5.sign()       
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()
		
		#x = p1.sign() + p2.sign() + p3.sign() + p4.sign() + p5.sign()
		x = (p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) 
		#x = p1 + p2 + p3 + p4 + p5
		
		##########################################################################		
		
		p1 = 25*model.module.bin_conv4_1(x[:,0:56,:,:])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
		
		p2 = 25*model.module.bin_conv4_2(x[:,56:112,:,:])
		p2_0 = p2.sign()       
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		
		p3 = 25*model.module.bin_conv4_3(x[:,112:168,:,:])
		p3_0 = p3.sign()       
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		
		p4 = 25*model.module.bin_conv4_4(x[:,168:224,:,:])
		p4_0 = p4.sign()       
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		
		p5 = 25*model.module.bin_conv4_5(x[:,224:280,:,:])
		p5_0 = p5.sign()       
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()
		
		p6 = 25*model.module.bin_conv4_6(x[:,280:336,:,:])
		p6_0 = p6.sign()       
		p6_N = (p6 + 512/args.ref_relative_distance).sign()
		p6_P = (p6 - 512/args.ref_relative_distance).sign()
		
		p7 = 25*model.module.bin_conv4_7(x[:,336:384,:,:])
		p7_0 = p7.sign()       
		p7_N = (p7 + 512/args.ref_relative_distance).sign()
		p7_P = (p7 - 512/args.ref_relative_distance).sign()
		
		#x = p1.sign() + p2.sign() + p3.sign() + p4.sign() + p5.sign() + p6.sign() + p7.sign()
		x = (p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) + (p6_N+p6_P) + (p7_N+p7_P) 
		#x = p1 + p2 + p3 + p4 + p5 + p6 + p7
		
	
		#####################################################################################
		#####################################################################################
		
		p1 = 25*model.module.bin_conv5_1(x[:,0:56,:,:])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
		
		p2 = 25*model.module.bin_conv5_2(x[:,56:112,:,:])
		p2_0 = p2.sign()       
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		
		p3 = 25*model.module.bin_conv5_3(x[:,112:168,:,:])
		p3_0 = p3.sign()       
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		
		p4 = 25*model.module.bin_conv5_4(x[:,168:224,:,:])
		p4_0 = p4.sign()       
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		
		p5 = 25*model.module.bin_conv5_5(x[:,224:280,:,:])
		p5_0 = p5.sign()       
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()
		
		p6 = 25*model.module.bin_conv5_6(x[:,280:336,:,:])
		p6_0 = p6.sign()       
		p6_N = (p6 + 512/args.ref_relative_distance).sign()
		p6_P = (p6 - 512/args.ref_relative_distance).sign()
		
		p7 = 25*model.module.bin_conv5_7(x[:,336:384,:,:])
		p7_0 = p7.sign()       
		p7_N = (p7 + 512/args.ref_relative_distance).sign()
		p7_P = (p7 - 512/args.ref_relative_distance).sign()
		
		#x = p1.sign() + p2.sign() + p3.sign() + p4.sign() + p5.sign() + p6.sign() + p7.sign()
		x = (p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) + (p6_N+p6_P) + (p7_N+p7_P) 
		#x = p1 + p2 + p3 + p4 + p5 + p6 + p7
				
		
		##########################################################################################
		##########################################################################################
		
		x = model.module.pool3(x)
		x = model.module.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		
		
		p1 = 25*model.module.bin_conv6_1(x[:,0:512])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
		
		p2 = 25*model.module.bin_conv6_2(x[:,512:1024])
		p2_0 = p2.sign()       
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		
		p3 = 25*model.module.bin_conv6_3(x[:,1024:1536])
		p3_0 = p3.sign()       
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		
		p4 = 25*model.module.bin_conv6_4(x[:,1536:2048])
		p4_0 = p4.sign()       
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		
		p5 = 25*model.module.bin_conv6_5(x[:,2048:2560])
		p5_0 = p5.sign()       
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()
		
		p6 = 25*model.module.bin_conv6_6(x[:,2560:3072])
		p6_0 = p6.sign()       
		p6_N = (p6 + 512/args.ref_relative_distance).sign()
		p6_P = (p6 - 512/args.ref_relative_distance).sign()
		
		p7 = 25*model.module.bin_conv6_7(x[:,3072:3584])
		p7_0 = p7.sign()       
		p7_N = (p7 + 512/args.ref_relative_distance).sign()
		p7_P = (p7 - 512/args.ref_relative_distance).sign()
		
		p8 = 25*model.module.bin_conv6_8(x[:,3584:4096])
		p8_0 = p8.sign()       
		p8_N = (p8 + 512/args.ref_relative_distance).sign()
		p8_P = (p8 - 512/args.ref_relative_distance).sign()
		
		p9 = 25*model.module.bin_conv6_9(x[:,4096:4608])
		p9_0 = p9.sign()       
		p9_N = (p9 + 512/args.ref_relative_distance).sign()
		p9_P = (p9 - 512/args.ref_relative_distance).sign()
		
		p10 = 25*model.module.bin_conv6_10(x[:,4608:5120])
		p10_0 = p10.sign()       
		p10_N = (p10 + 512/args.ref_relative_distance).sign()
		p10_P = (p10 - 512/args.ref_relative_distance).sign()
		
		p11 = 25*model.module.bin_conv6_11(x[:,5120:5632])
		p11_0 = p11.sign()       
		p11_N = (p11 + 512/args.ref_relative_distance).sign()
		p11_P = (p11 - 512/args.ref_relative_distance).sign()
		
		p12 = 25*model.module.bin_conv6_12(x[:,5632:6144])
		p12_0 = p12.sign()       
		p12_N = (p12 + 512/args.ref_relative_distance).sign()
		p12_P = (p12 - 512/args.ref_relative_distance).sign()
		
		p13 = 25*model.module.bin_conv6_13(x[:,6144:6656])
		p13_0 = p13.sign()       
		p13_N = (p13 + 512/args.ref_relative_distance).sign()
		p13_P = (p13 - 512/args.ref_relative_distance).sign()
		
		
		p14 = 25*model.module.bin_conv6_14(x[:,6656:7168])
		p14_0 = p14.sign()       
		p14_N = (p14 + 512/args.ref_relative_distance).sign()
		p14_P = (p14 - 512/args.ref_relative_distance).sign()
		
		
		p15 = 25*model.module.bin_conv6_15(x[:,7168:7680])
		p15_0 = p15.sign()       
		p15_N = (p15 + 512/args.ref_relative_distance).sign()
		p15_P = (p15 - 512/args.ref_relative_distance).sign()
		
		p16 = 25*model.module.bin_conv6_16(x[:,7680:8192])
		p16_0 = p16.sign()       
		p16_N = (p16 + 512/args.ref_relative_distance).sign()
		p16_P = (p16 - 512/args.ref_relative_distance).sign()
		
		p17 = 25*model.module.bin_conv6_17(x[:,8192:8704])
		p17_0 = p17.sign()       
		p17_N = (p17 + 512/args.ref_relative_distance).sign()
		p17_P = (p17 - 512/args.ref_relative_distance).sign()
		
		p18 = 25*model.module.bin_conv6_18(x[:,8704:9216])
		p18_0 = p18.sign()       
		p18_N = (p18 + 512/args.ref_relative_distance).sign()
		p18_P = (p18 - 512/args.ref_relative_distance).sign()
		
		#x = p1.sign() + p2.sign() + p3.sign() + p4.sign() + p5.sign() + p6.sign() + p7.sign() + p8.sign() + p9.sign() + p10.sign() + p11.sign() + p12.sign() + p13.sign() + p14.sign() + p15.sign() + p16.sign() + p17.sign() + p18.sign()
		x = (p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) + (p6_N+p6_P) + (p7_N+p7_P) + (p8_N+p8_P)+ (p9_N+p9_P)+ (p10_N+p10_P)+ (p11_N+p11_P)+ (p12_N+p12_P)+ (p13_N+p13_P)+ (p14_N+p14_P)+ (p15_N+p15_P)+ (p16_N+p16_P)+ (p17_N+p17_P)+ (p18_N+p18_P)
		#x = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18	
		
		###########################################################################################
		###########################################################################################
		
		p1 = 40*model.module.bin_conv7_1(x[:,0:512])
		p1_0 = p1.sign()       
		p1_N = (p1 + 512/args.ref_relative_distance).sign()
		p1_P = (p1 - 512/args.ref_relative_distance).sign()
				
		p2 = 40*model.module.bin_conv7_2(x[:,512:1024])
		p2_0 = p2.sign() 
		p2_N = (p2 + 512/args.ref_relative_distance).sign()
		p2_P = (p2 - 512/args.ref_relative_distance).sign()
		
		p3 = 40*model.module.bin_conv7_3(x[:,1024:1536])
		p3_0 = p3.sign()        
		p3_N = (p3 + 512/args.ref_relative_distance).sign()
		p3_P = (p3 - 512/args.ref_relative_distance).sign()
		
		p4 = 40*model.module.bin_conv7_4(x[:,1536:2048])
		p4_0 = p4.sign()     
		p4_N = (p4 + 512/args.ref_relative_distance).sign()
		p4_P = (p4 - 512/args.ref_relative_distance).sign()
		
		p5 = 30*model.module.bin_conv7_5(x[:,2048:2560])
		p5_0 = p5.sign()     
		p5_N = (p5 + 512/args.ref_relative_distance).sign()
		p5_P = (p5 - 512/args.ref_relative_distance).sign()	
		
		p6 = 40*model.module.bin_conv7_6(x[:,2560:3072])
		p6_0 = p6.sign()     
		p6_N = (p6 + 512/args.ref_relative_distance).sign()
		p6_P = (p6 - 512/args.ref_relative_distance).sign()	
		
		p7 = 40*model.module.bin_conv7_7(x[:,3072:3584])
		p7_0 = p7.sign()     
		p7_N = (p7 + 512/args.ref_relative_distance).sign()
		p7_P = (p7 - 512/args.ref_relative_distance).sign()	
		
		p8 = 40*model.module.bin_conv7_8(x[:,3584:4096])
		p8_0 = p8.sign()     
		p8_N = (p8 + 512/args.ref_relative_distance).sign()
		p8_P = (p8 - 512/args.ref_relative_distance).sign()	
		
		#x = 20*(p1.sign() + p2.sign() + p3.sign() + p4.sign() + p5.sign() + p6.sign() + p7.sign() + p8.sign())
		x = ((p1_N+p1_P) + (p2_N+p2_P)  + (p3_N+p3_P)  + (p4_N+p4_P)  + (p5_N+p5_P) + (p6_N+p6_P) + (p7_N+p7_P) + (p8_N+p8_P)) 
		#x = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
				
		
		######################################################################################
		######################################################################################
		
		x = model.module.bn2(x)
		x = model.module.dropout(x)
		output = model.module.linear(x)
		######################################################################################
		######################################################################################
		######################################################################################
		
		
		loss = criterion(output, target)
	
		# measure accuracy and record loss
		test_loss += criterion(output, target).data.item()
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	
	bin_op.restore()
	acc = 100. * float(correct) / len(testloader.dataset)
	
	#file1.write(str(epoch) + ": " + str(acc) + "\n")
	#file1.flush()
       
	
	global best_acc
	if acc > best_acc:
		best_acc = acc
		#save_state(model, best_acc)
	
	test_loss /= len(testloader.dataset)
	
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format( test_loss * 128., correct, len(testloader.dataset), 100. * float(correct) / len(testloader.dataset)))
	print('Best Accuracy: {:.2f}%\n'.format(best_acc))	
	
	return 


def save_state(model, best_acc):
    
	print('==> Saving model ...')
	state = {
			'best_acc': best_acc,
			'state_dict': model.state_dict(),
			}
	#new_state_=state.copy()
	for key in list(state['state_dict'].keys()):
		if 'module' in key:
			state['state_dict'][key.replace('module.', '')] = \
					state['state_dict'].pop(key)
	#torch.save(state, 'models/'+args.arch+'.best.pth.tar')
	torch.save(state, 'models/'+'alexnet_BCIM_no_Relu_All_approximate'+'.best.pth.tar')
	
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')



def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	print ('Learning rate:', lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr




if __name__ == '__main__':
	main()
