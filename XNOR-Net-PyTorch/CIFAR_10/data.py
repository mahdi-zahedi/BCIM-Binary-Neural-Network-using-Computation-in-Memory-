import os
import torch
import pickle
import numpy
import torchvision
import torchvision.transforms as transforms

class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        
        batch_size = 5

        if self.train:
            
            trainset = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=self.transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            for (data, target) in trainloader:
                self.train_data = data
                self.train_labels = target
                #self.train_data = torch.from_numpy(self.train_data.astype('float32'))
                #self.train_labels = torch.from_numpy(self.train_labels.astype('int'))
        else:
        
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=self.transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
            for (data, target) in testloader:            
                self.test_data = data
                self.test_labels = target
                #self.test_data = torch.from_numpy(self.test_data.astype('float32'))
                #self.test_labels = torch.from_numpy(self.test_labels.astype('int'))
                
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target
