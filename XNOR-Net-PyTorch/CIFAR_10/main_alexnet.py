from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import time

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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
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
	file1 = open('acc_%s.txt' % args.arch, 'w')
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
		
	print(model)
	
	#############################################################################################
	#############################################################################################
	
	# define loss function (criterion) and optimizer
	#define solver and criterion
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
		test()
		exit(0)
	
	for epoch in range(1, args.epochs):
		adjust_learning_rate(optimizer, epoch)
	
		# train for one epoch
		train(model, epoch)
	
		# evaluate on validation set
		test(model, epoch)
	
		# remember best prec@1 and save checkpoint
		#is_best = prec1 > best_prec1
		#best_prec1 = max(prec1, best_prec1)
		#save_checkpoint({
		#    'epoch': epoch + 1,
		#    'arch': args.arch,
		#    'state_dict': model.state_dict(),
		#    'best_prec1': best_prec1,
		#    'optimizer' : optimizer.state_dict(),
		#}, is_best)


def train(model, epoch):
	#batch_time = AverageMeter()
	#data_time = AverageMeter()
	#losses = AverageMeter()
	#top1 = AverageMeter()
	#top5 = AverageMeter()
	
	# switch to train mode
	model.train()

    #end = time.time()
	for batch_idx, (data, target) in enumerate(trainloader):
		# measure data loading time
		#data_time.update(time.time() - end)
	
		data, target = Variable(data.cuda()), Variable(target.cuda())
	
		# process the weights including binarization
		#optimizer.zero_grad()
		bin_op.binarization()
		
		# compute output
		output = model(data)
		loss = criterion(output, target)
	
		# measure accuracy and record loss
		#prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		#losses.update(loss.data.item(), input.size(0))
		#top1.update(prec1[0], input.size(0))
		#top5.update(prec5[0], input.size(0))
	
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
	
		# restore weights
		bin_op.restore()
		bin_op.updateBinaryGradWeight()
	
		optimizer.step()
	
		# measure elapsed time
		#batch_time.update(time.time() - end)
		#end = time.time()
	
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
				epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.data.item(), optimizer.param_groups[0]['lr']))
		
	return


def test(model, epoch):
	
	#batch_time = AverageMeter()
	#losses = AverageMeter()
	#top1 = AverageMeter()
	#top5 = AverageMeter()
	
	# switch to evaluate mode
	model.eval()
	test_loss = 0
	correct = 0
	#end = time.time()
	bin_op.binarization()
	for data, target in testloader:
		data, target = Variable(data.cuda()), Variable(target.cuda())
	
		# compute output
		output = model(data)
		loss = criterion(output, target)
	
		# measure accuracy and record loss
		test_loss += criterion(output, target).data.item()
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	
	bin_op.restore()
	acc = 100. * float(correct) / len(testloader.dataset)
	
	file1.write(str(epoch) + ": " + str(acc) + "\n")
	file1.flush()
       # measure elapsed time
       #batch_time.update(time.time() - end)
       #end = time.time()
	
	global best_acc
	if acc > best_acc:
		best_acc = acc
		save_state(model, best_acc)
	
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
	torch.save(state, 'models/'+args.arch+'.best.pth.tar')
	
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
