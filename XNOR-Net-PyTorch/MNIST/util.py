import torch.nn as nn
import numpy
import pdb

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        
	#########################
        start_range = 1 # defult=1
        #########################
        end_range = count_targets-2 #defult=-2
        #########################
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        #print(self.bin_range)
        #pdb.set_trace()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        #print(self.bin_range)
        index = -1
        
        #######################
        #self.mymodel=model
        #######################
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        #print (self.mymodel.bin_conv1.conv.weight[:,:,:,:])
        #pdb.set_trace()
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
    
            if(negMean+self.target_modules[index].data!=0).all():
            	self.target_modules[index].data = self.target_modules[index].data.add(negMean)
        #print (self.mymodel.bin_conv1.conv.weight[:,:,:,:])
        #pdb.set_trace()

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            #print('sdsd')
            #print(n)
            #print(s)
            #pdb.set_trace()
            
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            
            ##############################Mahdi
            #print(m.size())
            #print(m)
            #print(self.target_modules[index])
            #print (m.expand(s))
            #pdb.set_trace()
            #print(self.target_modules[index].data.sign())
            #print(self.target_modules[index].data.sign().mul(m.expand(s)))
            #pdb.set_trace()
            ##############################
            
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign()#.mul(m.expand(s))
            ###############
            #print(self.target_modules[index].data)
            #pdb.set_trace()
        #print(self.target_modules)
        #pdb.set_trace()
            ##############	
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
