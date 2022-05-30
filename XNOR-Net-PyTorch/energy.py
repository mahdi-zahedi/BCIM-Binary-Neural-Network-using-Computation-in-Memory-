import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import energy_f
import argparse
import matplotlib.ticker as tick

if __name__=='__main__':


	#parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	#parser.add_argument('--arch', action='store', default='LeNet_5', help='the MNIST network structure: LeNet_5')
	#args = parser.parse_args()
	
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	
	##################################### LeNet_5_new_S ##################################################################
	##################################### LeNet_5_new_S ##################################################################
	
	#if args.arch == 'LeNet_5_new_S':
	
	   ################################ Layer1 #######################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 6
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	E_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi= energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	E_LeNet_5_new_S_L1 = E_SoTA
	
	################################### Layer2 #######################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 6
	output_channels = 16
	datatype_size = 8
	input_size_x = 12
	input_size_y = 12
	stride = 1
	
	E_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_SoTA  = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	E_LeNet_5_new_S_L2 = E_SoTA
	
	################################## Layer3 #########################
	input_channels = 16*4*4
	output_channels = 120
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	E_LeNet_5_new_S_L3 = E_SoTA
	################################## Layer4 #########################
       
	input_channels = 120
	output_channels = 84
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	E_LeNet_5_new_S_L4 = E_SoTA
	################################# Laye5 ############################
	input_channels = 84
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_LeNet5 = energy_SoTA 
	energy_Mahdi_LeNet5 = energy_Mahdi
	
	E_LeNet_5_new_S_L5 = E_SoTA
	
	
	
	##################################### CNN1 ##################################################################
	##################################### CNN1 ##################################################################
	
	#elif args.arch == 'CNN1':
	
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	   ################################## Layer1 ############################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 5
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	E_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi= energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	#print('CNN1')
	#print (E_SoTA)
	#print (E_Mahdi)
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	#print (E_SoTA)
	#print (E_Mahdi)
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	#print (E_SoTA)
	#print (E_Mahdi)
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_CNN1 = energy_SoTA 
	energy_Mahdi_CNN1 = energy_Mahdi
	#print (E_SoTA)
	#print (E_Mahdi)	
	

	##################################### CNN1_SA3 ##################################################################
	##################################### CNN1_SA3 ##################################################################
	
	#elif args.arch == 'CNN1':
	
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	   ################################## Layer1 ############################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 5
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	E_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi= energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_CNN1_SA3 = energy_SoTA 
	energy_Mahdi_CNN1_SA3 = energy_Mahdi
	
	
	##################################### CNN2 ##################################################################
	##################################### CNN2 ##################################################################	
	#elif args.arch == 'CNN2':
	
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	E_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi= energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	E_CNN2_L1 = E_SoTA
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	E_CNN2_L2 = E_SoTA
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	E_CNN2_L3 = E_SoTA
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_CNN2 = energy_SoTA 
	energy_Mahdi_CNN2 = energy_Mahdi
	
	E_CNN2_L4 = E_SoTA
	
	print (energy_SoTA)
	print (energy_Mahdi)
	##################################### CNN2_SA3 ##################################################################
	##################################### CNN2_SA3 ##################################################################	
	#elif args.arch == 'CNN2':
	
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	E_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi= energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_CNN2_SA3 = energy_SoTA 
	energy_Mahdi_CNN2_SA3 = energy_Mahdi
	print (energy_SoTA)
	print (energy_Mahdi)
	##################################### MLP-S ##################################################################
	##################################### MLP-S ##################################################################
	#elif args.arch == 'MLP-S':
		
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	############################## Layer1 ###########################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	############################## Layer2 ##########################
       
	input_channels = 784
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	############################ Layer3 #############################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	######################### Layer4 #################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_MLP_S = energy_SoTA 
	energy_Mahdi_MLP_S = energy_Mahdi
		
		
		
		
		
	#elif args.arch == 'MLP-M':
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1000
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer3 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer4 ##################################################################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer5 ##################################################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_MLP_M = energy_SoTA 
	energy_Mahdi_MLP_M = energy_Mahdi
	
	
	
	
	#elif args.arch == 'MLP-L':
	energy_SoTA = 0
	energy_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1500
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer3 ##################################################################
       
	input_channels = 1500
	output_channels = 1000
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer4 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	##################################### Layer5 ##################################################################
	input_channels = 500
	output_channels = 10
	datatype_size = 8
	
	E_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	energy_SoTA_MLP_L = energy_SoTA 
	energy_Mahdi_MLP_L = energy_Mahdi
	
	#else:
	#	print('ERROR: specified arch is not suppported')
	#	exit()
	
	
	#print (energy_SoTA_LeNet5)
	#print (energy_Mahdi_LeNet5)
	barWidth = 0.3
 
	F1=[energy_SoTA_LeNet5/energy_SoTA_LeNet5,energy_SoTA_MLP_S/energy_SoTA_MLP_S,energy_SoTA_MLP_M/energy_SoTA_MLP_M,energy_SoTA_MLP_L/energy_SoTA_MLP_L, energy_SoTA_CNN1/energy_SoTA_CNN1, energy_SoTA_CNN2/energy_SoTA_CNN2]
	F2=[energy_SoTA_LeNet5/energy_Mahdi_LeNet5,energy_SoTA_MLP_S/energy_Mahdi_MLP_S,energy_SoTA_MLP_M/energy_Mahdi_MLP_M,energy_SoTA_MLP_L/energy_Mahdi_MLP_L,energy_SoTA_CNN1/energy_Mahdi_CNN1,energy_SoTA_CNN2/energy_Mahdi_CNN2]
	F3=[0,0,0,0,energy_SoTA_CNN1/energy_Mahdi_CNN1_SA3,energy_SoTA_CNN2/energy_Mahdi_CNN2_SA3]
	
	r1 = np.arange(len(F1))
	#r2 = [x + barWidth for x in r1]
	#r2 = np.arange(len(F2))
	#r3 = np.arange(len(F3))
	
	fig, ax = plt.subplots()
	
	plt.bar(r1-barWidth, F1, width = barWidth, color = 'darkolivegreen', edgecolor = 'black', capsize=7, label='Baseline', hatch='//')
	
	plt.bar(r1, F2, width = barWidth, color = 'olivedrab', edgecolor = 'black', capsize=7, label='Proposed with 1 Ref', hatch='//')
	plt.bar(r1+barWidth, F3, width = barWidth, capsize=7, color = 'honeydew', edgecolor = 'black', label='Proposed with 3 Ref', hatch='//')
	
	plt.xticks([r for r in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2'])
	
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:}x'.format(x) for x in vals])
	plt.ylabel('Energy improvement',fontsize=15)
	plt.xticks(rotation=45)
	#ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	#ylim(0, 400)
	plt.tight_layout() 
	plt.legend()
	fig.set_size_inches(6, 3)
	plt.savefig("energy.svg")
	
	#######################################################################################
	#######################################################################################
	
	fig, ax = plt.subplots()
	fig.set_figheight(3)
	fig.set_figwidth(6)
	colors1 = ( "silver","wheat", "olive", "maroon", "teal")
	colors2 = ( "orange", "cyan", "brown","grey")
	ax1 = plt.subplot(1, 2, 1)
	y1 = np.array([E_LeNet_5_new_S_L1, E_LeNet_5_new_S_L2, E_LeNet_5_new_S_L3, E_LeNet_5_new_S_L4,E_LeNet_5_new_S_L5])
	mylabels = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]
	myexplode = [0.1, 0, 0, 0, 0]
	
	#ax1.pie(y1, labels = mylabels, explode = myexplode, colors=colors1)
	patches, texts = plt.pie(y1, explode = myexplode, colors=colors1)
	ax1.set_title("LeNet_5")
	plt.legend(patches, mylabels, loc="best")
	
	
	ax2 = plt.subplot(1, 2, 2)
	y2 = np.array([E_CNN2_L1, E_CNN2_L2, E_CNN2_L3, E_CNN2_L4])
	mylabels2 = ["Layer1", "Layer2", "Layer3", "Layer4"]
	myexplode2 = [0.1, 0, 0, 0]
	
	#ax2.pie(y2, labels = mylabels2, explode = myexplode2, colors=colors1)
	patches, texts = plt.pie(y2, explode = myexplode2, colors=colors1)
	ax2.set_title("CNN2")
	plt.legend(patches, mylabels2, loc="best")
	plt.savefig("energy_pie.svg")
	
	