from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb
import util
import models

#from models import nin
#from models import nin_without_bn
#from models import nin_without_bn_non_seq
from torch.autograd import Variable
 

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
	
	#print('==> Saving model ...')
    #state = {
    #        'best_acc': best_acc,
    #        'state_dict': model.state_dict(),
    #        }
    #for key in state['state_dict'].keys():
    #    if 'module' in key:
    #        state['state_dict'][key.replace('module.', '')] = \
    #                state['state_dict'].pop(key)
    #torch.save(state, 'models/'+args.arch+'.best.pth.tar')
	
def train(epoch):
    #pdb.set_trace()
    model.train()
    #pdb.set_trace()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        
        #print(type(trainloader))
        #print(list(enumerate(trainloader)))
        #print(batch_idx)
        #print(data.size())
        #print(target.size())
        #pdb.set_trace()
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        bin_op.binarization()
		
        #for m in model.modules():  
        #    print(m)
        #    print(m.weight.size())
            
        #print (vars(model))
        #pdb.set_trace()
        output = model(data)
        output = output[:, :, 0,0]
        
        # backwarding
        #print(output.size())
        #print(target.size())
        loss = criterion(output, target)
        loss.backward()
        
        """
        print(model)
        pdb.set_trace()
        for a in model.modules():
            if isinstance(a, nn.Conv2d):
                print(a)
                print(a.weight)
                pdb.set_trace()
        """		
		
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        #data, target = Variable(data), Variable(target)
                                    
        output = model(data)
        output = output[:, :, 0,0]
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)
	
    file1.write(str(epoch) + ": " + str(acc) + "\n")
    file1.flush()
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    #if not os.path.isfile(args.data+'/train_data'):
        # check the data path
    #    raise Exception\
    #            ('Please assign the correct data path with --data <DATA_PATH>')


    ##################################################################
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4 

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    ##################################################################
    
    #trainset = data.dataset(root=args.data, train=True)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #        shuffle=True, num_workers=2)

    #testset = data.dataset(root=args.data, train=False)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #        shuffle=False, num_workers=2)
    ###################################################################
    
    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #######################################################################
    file1 = open('acc_%s.txt' % args.arch, 'w')
    #######################################################################	
    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin_non_seq':
        model = models.nin_non_seq()	
    elif args.arch == 'nin_without_bn':
        model = models.nin_without_bn()			
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

    optimizer = optim.Adam(params, lr=base_lr,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    #pdb.set_trace()
    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)
			
    # start training
    for epoch in range(1, 500):
        #adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
