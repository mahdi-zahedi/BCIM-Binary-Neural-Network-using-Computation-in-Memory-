import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import latency_f
import argparse
import matplotlib.ticker as tick

if __name__=='__main__':


	#parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	#parser.add_argument('--arch', action='store', default='LeNet_5', help='the MNIST network structure: LeNet_5')
	#args = parser.parse_args()
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0
	
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
	layer_name = 'LeNet5_Layer1'
	
	L_SoTA = latency_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
					   	
	L_Mahdi= latency_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
						   
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	L_LeNet_5_new_S_L1 = L_SoTA
	L_LeNet_5_new_S_L1_Mahdi = L_Mahdi
	################################### Layer2 #######################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 6
	output_channels = 16
	datatype_size = 8
	input_size_x = 12
	input_size_y = 12
	stride = 1
	layer_name = 'LeNet5_Layer2'
	L_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_SoTA  = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	L_LeNet_5_new_S_L2 = L_SoTA
	L_LeNet_5_new_S_L2_Mahdi = L_Mahdi
	################################## Layer3 #########################
	input_channels = 16*4*4
	output_channels = 120
	datatype_size = 8
	layer_name = 'LeNet5_Layer3'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)						 
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	print('L_LeNet_5_new_S_L3')
	print(L_Mahdi)
	print(L_SoTA)
	
	L_LeNet_5_new_S_L3 = L_SoTA
	L_LeNet_5_new_S_L3_Mahdi = L_Mahdi
	################################## Layer4 #########################
       
	input_channels = 120
	output_channels = 84
	datatype_size = 8
	layer_name = 'LeNet5_Layer4'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	L_LeNet_5_new_S_L4 = L_SoTA
	L_LeNet_5_new_S_L4_Mahdi = L_Mahdi
	################################# Layer5 ############################
	input_channels = 84
	output_channels = 10
	datatype_size = 8
	layer_name = 'LeNet5_Layer5'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)					   
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_LeNet5 = Latency_SoTA 
	Latency_Mahdi_LeNet5 = Latency_Mahdi
	
	L_LeNet_5_new_S_L5 = L_SoTA
	L_LeNet_5_new_S_L5_Mahdi = L_Mahdi
	
	
	##################################### CNN1 ##################################################################
	##################################### CNN1 ##################################################################
	
	#elif args.arch == 'CNN1':
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
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
	layer_name = 'CNN1_Layer1'
	
	L_SoTA = latency_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)				   	
	L_Mahdi= latency_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
					   
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	#print('CNN1')
	#print (L_SoTA)
	#print (L_Mahdi)
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	layer_name = 'CNN1_Layer2'
	
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)				 
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)

					   
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	#print (L_SoTA)
	#print (L_Mahdi)
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	layer_name = 'CNN1_Layer3'
	
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)					    	
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	#print (L_SoTA)
	#print (L_Mahdi)
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	layer_name = 'CNN1_Layer4'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)					   
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_CNN1 = Latency_SoTA 
	Latency_Mahdi_CNN1 = Latency_Mahdi
	#print (L_SoTA)
	#print (L_Mahdi)	
	
	
	##################################### CNN1_SA3 ##################################################################
	##################################### CNN1_SA3 ##################################################################
	
	#elif args.arch == 'CNN1':
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0
	   ################################## Layer1 ############################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 5
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	layer_name = 'CNN1_SA3_Layer1'
	L_SoTA = latency_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)					   
	
	L_Mahdi= latency_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
					   
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer2'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels)	                     
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer3'
	
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels)	
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer4'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
					   
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_CNN1_SA3 = Latency_SoTA 
	Latency_Mahdi_CNN1_SA3 = Latency_Mahdi
	
	
	##################################### CNN2 ##################################################################
	##################################### CNN2 ##################################################################	
	#elif args.arch == 'CNN2':
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	layer_name = 'CNN2_Layer1'
	L_SoTA = latency_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
	L_Mahdi= latency_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	L_CNN2_L1 = L_SoTA
	L_CNN2_L1_Mahdi = L_Mahdi
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_Layer2'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )	
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	L_CNN2_L2 = L_SoTA
	L_CNN2_L2_Mahdi = L_Mahdi
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_Layer3'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	L_CNN2_L3 = L_SoTA
	L_CNN2_L3_Mahdi = L_Mahdi
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	layer_name = 'CNN2_Layer4'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_CNN2 = Latency_SoTA 
	Latency_Mahdi_CNN2 = Latency_Mahdi
	
	L_CNN2_L4 = L_SoTA
	L_CNN2_L4_Mahdi = L_Mahdi
	##################################### CNN2_SA3 ##################################################################
	##################################### CNN2_SA3 ##################################################################	
	#elif args.arch == 'CNN2':
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	layer_name = 'CNN2_SA3.Layer1'
	L_SoTA = latency_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name )
	L_Mahdi= latency_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name )
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_SA3.Layer2'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	layer_name = 'CNN2_SA3.Layer3'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	layer_name = 'CNN2_SA3.Layer4'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_CNN2_SA3 = Latency_SoTA 
	Latency_Mahdi_CNN2_SA3 = Latency_Mahdi
	
	##################################### MLP-S ##################################################################
	##################################### MLP-S ##################################################################
	#elif args.arch == 'MLP-S':
		
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0
	############################## Layer1 ###########################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-S_Layer1'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	############################## Layer2 ##########################
       
	input_channels = 784
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-S_Layer2'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	############################ Layer3 #############################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	layer_name = 'MLP-S_Layer3'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	######################### Layer4 #################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-S_Layer4'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_MLP_S = Latency_SoTA 
	Latency_Mahdi_MLP_S = Latency_Mahdi
		
		
		
		
		
	#elif args.arch == 'MLP-M':
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-M_Layer1'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1000
	datatype_size = 8
	layer_name = 'MLP-M_Layer2'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer3 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-M_Layer3'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer4 ##################################################################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	layer_name = 'MLP-M_Layer4'
	
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_SoTA = Latency_SoTA + L_Mahdi
	##################################### Layer5 ##################################################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-M_Layer5'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_MLP_M = Latency_SoTA 
	Latency_Mahdi_MLP_M = Latency_Mahdi
	
	
	
	
	#elif args.arch == 'MLP-L':
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	L_SoTA = 0
	L_Mahdi = 0	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-L_Layer1'
	L_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1500
	datatype_size = 8
	layer_name = 'MLP-L_Layer2'
	
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer3 ##################################################################
       
	input_channels = 1500
	output_channels = 1000
	datatype_size = 8
	layer_name = 'MLP-L_Layer3'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer4 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-L_Layer4'
	L_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	##################################### Layer5 ##################################################################
	input_channels = 500
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-L_Layer5'
	L_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Latency_SoTA_MLP_L = Latency_SoTA 
	Latency_Mahdi_MLP_L = Latency_Mahdi
	
	#else:
	#	print('ERROR: specified arch is not suppported')
	#	exit()
	
	
	#print (Latency_SoTA_LeNet5)
	#print (Latency_Mahdi_LeNet5)
	barWidth = 0.3
 
	F1=[Latency_SoTA_LeNet5/Latency_SoTA_LeNet5,Latency_SoTA_MLP_S/Latency_SoTA_MLP_S,Latency_SoTA_MLP_M/Latency_SoTA_MLP_M,Latency_SoTA_MLP_L/Latency_SoTA_MLP_L, Latency_SoTA_CNN1/Latency_SoTA_CNN1, Latency_SoTA_CNN2/Latency_SoTA_CNN2]
	F2=[Latency_SoTA_LeNet5/Latency_Mahdi_LeNet5,Latency_SoTA_MLP_S/Latency_Mahdi_MLP_S,Latency_SoTA_MLP_M/Latency_Mahdi_MLP_M,Latency_SoTA_MLP_L/Latency_Mahdi_MLP_L,Latency_SoTA_CNN1/Latency_Mahdi_CNN1,Latency_SoTA_CNN2/Latency_Mahdi_CNN2]
	F3=[0,0,0,0,Latency_SoTA_CNN1/Latency_Mahdi_CNN1_SA3,Latency_SoTA_CNN2/Latency_Mahdi_CNN2_SA3]
	
	r1 = np.arange(len(F1))
	#r2 = [x + barWidth for x in r1]
	#r2 = np.arange(len(F2))
	#r3 = np.arange(len(F3))
	
	fig, ax = plt.subplots()
	
	plt.bar(r1-barWidth, F1, width = barWidth, color = 'navy', edgecolor = 'black', capsize=7, label='Baseline', hatch='//')
	
	plt.bar(r1, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=7, label='Proposed with 1 Ref', hatch='//')
	plt.bar(r1+barWidth, F3, width = barWidth, capsize=7, color = 'lightsteelblue', edgecolor = 'black', label='Proposed with 3 Ref', hatch='//')
	
	plt.xticks([r for r in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2'])
	
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:}x'.format(x) for x in vals])
	plt.ylabel('Latency improvement',fontsize=15)
	plt.xticks(rotation=45)
	#ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	#ylim(0, 400)
	plt.tight_layout() 
	plt.legend()
	fig.set_size_inches(6, 3)
	ax.set_yscale('log')
	plt.savefig("Latency.svg")
	
	#######################################################################################
	#######################################################################################
	#fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
	
	#colors1 = ( "navy", "olive", "teal","silver", "maroon")
	colors1 = ( "silver","navy", "olive", "maroon", "teal")
	colors2 = ( "orange", "cyan", "brown","grey")
	
	fig = plt.figure()
	fig.set_figheight(4)
	fig.set_figwidth(8)
	
	ax = plt.subplot(1, 2, 1)
	y1 = np.array([L_CNN2_L1/L_CNN2_L1_Mahdi, L_CNN2_L2/L_CNN2_L2_Mahdi, L_CNN2_L3/L_CNN2_L3_Mahdi, L_CNN2_L4/L_CNN2_L4_Mahdi])
	mylabels = ["Layer1", "Layer2", "Layer3", "Layer4"]
	r1 = np.arange(len(mylabels))
	#myexplode = [0.1, 0, 0, 0, 0]
	vals = ax.get_yticks()
	#ax.set_yticklabels(['{:}x'.format(x) for x in vals])
	plt.bar(mylabels, y1, width = barWidth, color = 'grey', edgecolor = 'black', capsize=7, hatch='//')
	plt.xticks([r for r in range(len(y1))], mylabels)
	ax.set_yscale('log')
	
	plt.ylabel('Latency improvement',fontsize=15)
	
	ax = plt.subplot(1, 2, 2)
	y2 = np.array([L_CNN2_L1, L_CNN2_L2, L_CNN2_L3, L_CNN2_L4])
	mylabels2 = ["Layer1", "Layer2", "Layer3", "Layer4"]
	myexplode2 = [0.1, 0, 0, 0]
	
	patches, texts = plt.pie(y2, explode = myexplode2, colors=colors1)
	plt.legend(patches, mylabels2, loc="best")
	ax.set_title("CNN2 latency per layer")
	plt.tight_layout() 
	plt.savefig("Latency_pie.svg")
	
	#######################################################################################
	#######################################################################################
	
	fig, ax = plt.subplots()
	colors1 = ( "silver","wheat", "olive", "maroon", "teal")
	colors2 = ( "orange", "cyan", "brown","grey")
	fig.set_figheight(4)
	fig.set_figwidth(8)
	
	
	ax = plt.subplot(1, 2, 1)
	y1 = np.array([L_LeNet_5_new_S_L1/L_LeNet_5_new_S_L1_Mahdi, L_LeNet_5_new_S_L2/L_LeNet_5_new_S_L2_Mahdi, L_LeNet_5_new_S_L3/L_LeNet_5_new_S_L3_Mahdi, L_LeNet_5_new_S_L4/L_LeNet_5_new_S_L4_Mahdi, L_LeNet_5_new_S_L5/L_LeNet_5_new_S_L5_Mahdi])
	mylabels = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]
	r1 = np.arange(len(mylabels))
	#myexplode = [0.1, 0, 0, 0, 0]
	vals = ax.get_yticks()
	#ax.set_yticklabels(['{:}x'.format(x) for x in vals])
	plt.bar(mylabels, y1, width = barWidth, color = 'grey', edgecolor = 'black', capsize=7, hatch='//')
	plt.xticks([r for r in range(len(y1))], mylabels, fontsize=12)
	ax.set_yscale('log')
	plt.ylabel('Latency improvement',fontsize=15)
	
	ax = plt.subplot(1, 2, 2)
	y2 = np.array([L_LeNet_5_new_S_L1, L_LeNet_5_new_S_L2, L_LeNet_5_new_S_L3, L_LeNet_5_new_S_L4, L_LeNet_5_new_S_L5])
	mylabels2 = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]
	myexplode2 = [0.1, 0, 0, 0, 0]
	
	patches, texts = plt.pie(y2, explode = myexplode2, colors=colors1)
	plt.legend(patches, mylabels2, loc="best")
	ax.set_title("LeNet5 latency per layer")
	plt.tight_layout() 
	plt.savefig("Latency_pie_LeNet5.svg")