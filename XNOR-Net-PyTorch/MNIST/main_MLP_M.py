from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import models
import util
import pdb
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy.random import seed
from numpy.random import randint
# seed random number generator
seed(2)
import util

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        
        bin_op.binarization()
        
        
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    return

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    
    bin_op.binarization()
    
    
    print(model)
    #pdb.set_trace()
    
    
    #print((model.bin_FC2.linear.weight.size()))
    
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
		###############################################################
		
        x = model.conv1(data)
        
		
		################################################################
        p1 = model.bin_FC2_1(x[:,0:392,:,:])
        p1=p1.sign()
        p2 = model.bin_FC2_2(x[:,392:784,:,:])
        p2=p2.sign()
             
        y=torch.zeros(p1.size(0), p1.size(1))
		
        for i in range(p1.size(0)):
            for j in range(p1.size(1)):                    
                if (p1[i][j]>0 and p2[i][j] > 0):
                    y[i][j]=1
                elif (p1[i][j]<0 and p2[i][j] < 0):
                    y[i][j]=-1
                else:
                    y[i][j]=-1
        #x = p1+p2
        x = y
		
        ################################################################
        p1 = model.bin_FC3_1(x[:,0:500])
        p1=p1.sign()
        p2 = model.bin_FC3_2(x[:,500:1000])
        p2=p2.sign()
             
        y=torch.zeros(p1.size(0), p1.size(1))
		
        for i in range(p1.size(0)):
            for j in range(p1.size(1)):                    
                if (p1[i][j]>0 and p2[i][j] > 0):
                    y[i][j]=1
                elif (p1[i][j]<0 and p2[i][j] < 0):
                    y[i][j]=-1
                else:
                    y[i][j]=-1
        #x = p1+p2
        x = y
		
        ################################################################
        #x = model.bin_FC2(x)
        x = model.bin_FC4(x)
        output = model.FC5(x)
		
		###################################
        #output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    
    ########################Mahdi
    #bin_op.restore()
    
    acc = 100. * float(correct) / len(test_loader.dataset)
    
    ########Mahdi
    file1.write(str(acc) + "\n")
    file1.flush()
	
    #save_state(model, acc)
    #############
    if (acc > best_acc):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)
			
    ##############
    bin_op.restore()
    ##############
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    file1 = open('acc_%s.txt' % args.arch, 'w')
	
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    elif args.arch == 'LeNet_5_without_bn': 
    	model = models.LeNet_5_without_bn()
    elif args.arch == 'LeNet_5_without_bn_Binary_in':
    	model = models.LeNet_5_without_bn_Binary_in()
    elif args.arch == 'LeNet_5_new_S':
	    model = models.LeNet_5_new_S()
    elif args.arch == 'MLP_S':
	    model = models.MLP_S()
    elif args.arch == 'MLP_M':
	    model = models.MLP_M()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
    
    
    #print(model)
    
    param_dict = dict(model.named_parameters())
    params = []
    
    base_lr = 0.1
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = optim.Adam(params, lr=args.lr,
            weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    if args.evaluate:
        test(evaluate=True)
        exit()
    #####
    state = {
            'acc': 0.0,
            'state_dict': model.state_dict(),
            }
    #####
    print(model)
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
    
