from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
import pdb
from zipfile import ZipFile
import pickle
import json

###############################################################

parser = argparse.ArgumentParser(description='Plot MNIST')
parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: 60)')

parser.add_argument('--arch', action='store', default='LeNet_5', help='the MNIST network structure: LeNet_5', type=str)

args = parser.parse_args()

arch_list = [item for item in args.arch.split(',')]

y=[[] for y in range(len(arch_list))] 
j=0
for x in arch_list:
	PATH=x+'.txt'
	f = open(PATH, "r")
	for line in f:
		y[j].append([float(x) for x in line.split()])		
	j=j+1
	
x=[]	
for epoch in range(1, args.epochs + 1):
	x.append(epoch)

##############################################################
print(y[0])
for idx in range(len(arch_list)):
	print('yes')
	print(len(y[idx]))
	plt.plot(x, y[idx], label=arch_list[idx])

plt.xlabel('Number of epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.locator_params(integer=True)
plt.axis([1, args.epochs, 10, 100])
leg = plt.legend();
plt.savefig("mygraph.pdf")
plt.show()

