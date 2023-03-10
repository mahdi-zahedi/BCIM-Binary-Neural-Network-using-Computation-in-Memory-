import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import energy_f
import argparse
import matplotlib.ticker as tick
from matplotlib import gridspec

if __name__=='__main__':
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	##################################### LeNet_5_new_S ##################################################################
	##################################### LeNet_5_new_S ##################################################################
	
	   ################################ Layer1 #######################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 6
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA, SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	###########################################
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	E_LeNet_5_new_S_L1 = E_SoTA
	
	################################### Layer2 #######################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 6
	output_channels = 16
	datatype_size = 1
	input_size_x = 12
	input_size_y = 12
	stride = 1
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC,T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_SoTA, T_SoTA, SA_SoTA  = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	E_LeNet_5_new_S_L2 = E_SoTA
	
	################################## Layer3 #########################
	input_channels = 16*4*4
	output_channels = 120
	datatype_size = 1
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	E_LeNet_5_new_S_L3 = E_SoTA
	
	################################## Layer4 #########################
       
	input_channels = 120
	output_channels = 84
	datatype_size = 1
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	E_LeNet_5_new_S_L4 = E_SoTA
	
	################################# Laye5 ############################
	input_channels = 84
	output_channels = 10
	datatype_size = 8
	
	E_SoTA,T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	
	energy_SoTA_LeNet5 = energy_SoTA 
	energy_Mahdi_LeNet5 = energy_Mahdi
	energy_Mahdi_ADC_LeNet5 = energy_Mahdi_ADC
	
	Transaction_SoTA_LeNet5 = Transactions_SoTA 
	Transaction_Mahdi_LeNet5 = Transactions_Mahdi
	
	
	SA_activation_SoTA_LeNet5 = SA_activations_SoTA 
	SA_activation_Mahdi_LeNet5 = SA_activations_Mahdi
			
	E_LeNet_5_new_S_L5 = E_SoTA
	
	
	##################################### CNN1 ##################################################################
	##################################### CNN1 ##################################################################
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	   ################################## Layer1 ############################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 5
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA, SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	###########################################
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	E_SoTA,T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi,T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	
	energy_SoTA_CNN1 = energy_SoTA 
	energy_Mahdi_CNN1 = energy_Mahdi
	energy_Mahdi_CNN1_ADC = energy_Mahdi_ADC
	
	Transaction_SoTA_CNN1 = Transactions_SoTA 
	Transaction_Mahdi_CNN1 = Transactions_Mahdi
	
	SA_activation_SoTA_CNN1 = SA_activations_SoTA 
	SA_activation_Mahdi_CNN1 = SA_activations_Mahdi
	
	##################################### CNN1_SA3 ##################################################################
	##################################### CNN1_SA3 ##################################################################
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	   ################################## Layer1 ############################
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 1
	output_channels = 5
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA, SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	###########################################
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	E_SoTA, T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	
	energy_SoTA_CNN1_SA3 = energy_SoTA 
	energy_Mahdi_CNN1_SA3 = energy_Mahdi
	
	##################################### CNN2 ##################################################################
	##################################### CNN2 ##################################################################	
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA, SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	###########################################
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	E_CNN2_L1 = E_SoTA
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	E_CNN2_L2 = E_SoTA
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	E_CNN2_L3 = E_SoTA
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	E_SoTA,T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi,T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	energy_SoTA_CNN2 = energy_SoTA 
	energy_Mahdi_CNN2 = energy_Mahdi
	energy_Mahdi_CNN2_ADC = energy_Mahdi_ADC
	
	Transaction_SoTA_CNN2 = Transactions_SoTA 
	Transaction_Mahdi_CNN2 = Transactions_Mahdi
	
	SA_activation_SoTA_CNN2 = SA_activations_SoTA 
	SA_activation_Mahdi_CNN2 = SA_activations_Mahdi
	
	E_CNN2_L4 = E_SoTA
	
	##################################### CNN2_SA3 ##################################################################
	##################################### CNN2_SA3 ##################################################################	
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	   ############################### Layer1 #################################
	kernel_size_x = 7
	kernel_size_y = 7
	input_channels = 1
	output_channels = 10
	datatype_size = 8
	input_size_x = 28
	input_size_y = 28
	stride = 1
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA, SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	###########################################
	
	#print(E_SoTA)
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	E_SoTA,T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi,SA_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_CNN2_SA3 = energy_SoTA 
	energy_Mahdi_CNN2_SA3 = energy_Mahdi
	
	Transaction_SoTA_CNN2 = Transactions_SoTA 
	Transaction_Mahdi_CNN2 = Transactions_Mahdi
	
	SA_activation_SoTA_CNN2_SA3 = SA_activations_SoTA 
	SA_activation_Mahdi_CNN2_SA3 = SA_activations_Mahdi
	

	##################################### MLP-S ##################################################################
	##################################### MLP-S ##################################################################
		
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	############################## Layer1 ###########################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	############################## Layer2 ##########################
       
	input_channels = 784
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	############################ Layer3 #############################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	######################### Layer4 #################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	
	E_SoTA, T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	
	energy_SoTA_MLP_S = energy_SoTA 
	energy_Mahdi_MLP_S = energy_Mahdi
	energy_Mahdi_MLP_S_ADC = energy_Mahdi_ADC	
		
	Transaction_SoTA_MLP_S = Transactions_SoTA 
	Transaction_Mahdi_MLP_S = Transactions_Mahdi
	
	SA_activation_SoTA_MLP_S = SA_activations_SoTA 
	SA_activation_Mahdi_MLP_S = SA_activations_Mahdi	
	

	#######################  MLP-M ###############################################################################
	##############################################################################################################
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0	
	E_Mahdi_ADC = 0
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi,T_Mahdi,SA_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC,T_Mahdi =  energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1000
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer3 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	##################################### Layer4 ##################################################################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer5 ##################################################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	
	E_SoTA, T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	energy_SoTA_MLP_M = energy_SoTA 
	energy_Mahdi_MLP_M = energy_Mahdi
	energy_Mahdi_MLP_M_ADC = energy_Mahdi_ADC	
		
	Transaction_SoTA_MLP_M = Transactions_SoTA 
	Transaction_Mahdi_MLP_M = Transactions_Mahdi
	
	SA_activation_SoTA_MLP_M = SA_activations_SoTA 
	SA_activation_Mahdi_MLP_M = SA_activations_Mahdi
		
	######################################MLP-L####################################################################
	######################################MLP-L####################################################################
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1500
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer3 ##################################################################
       
	input_channels = 1500
	output_channels = 1000
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer4 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	
	E_Mahdi, T_Mahdi, SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size)
	E_SoTA, T_SoTA, SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	##################################### Layer5 ##################################################################
	input_channels = 500
	output_channels = 10
	datatype_size = 8
	
	E_SoTA, T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	
	energy_SoTA_MLP_L = energy_SoTA 
	energy_Mahdi_MLP_L = energy_Mahdi
	energy_Mahdi_MLP_L_ADC = energy_Mahdi_ADC	
		
	Transaction_SoTA_MLP_L = Transactions_SoTA 
	Transaction_Mahdi_MLP_L = Transactions_Mahdi
	
	SA_activation_SoTA_MLP_L = SA_activations_SoTA 
	SA_activation_Mahdi_MLP_L = SA_activations_Mahdi
	
	
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	
	energy_SoTA = 0
	energy_Mahdi = 0
	energy_Mahdi_ADC = 0
	
	Transactions_SoTA = 0
	T_SoTA = 0
	SA_activations_SoTA = 0
	SA_SoTA = 0
	
	
	Transactions_Mahdi = 0
	SA_activations_Mahdi = 0
	T_Mahdi = 0
	SA_Mahdi = 0
	
	E_SoTA = 0
	E_Mahdi = 0
	E_Mahdi_ADC = 0
	
	##################################### Layer1 ##################################################################
	kernel_size_x = 11
	kernel_size_y = 11
	input_channels = 3
	output_channels = 96
	datatype_size = 8
	input_size_x = 256
	input_size_y = 256
	stride = 4
	
	#E_SoTA,T_SoTA = energy_f.CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	#E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	####################################################
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	energy_SoTA_AlexNet_L1 = E_SoTA
	energy_Mahdi_AlexNet_L1 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L1 = E_Mahdi_ADC 
	##################################### Layer2 ##################################################################
	
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 96
	output_channels = 256
	datatype_size = 8
	input_size_x = 31
	input_size_y = 31
	stride = 1
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L2 = E_SoTA
	energy_Mahdi_AlexNet_L2 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L2 = E_Mahdi_ADC
	##################################### Layer3 ##################################################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 256
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L3 = E_SoTA
	energy_Mahdi_AlexNet_L3 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L3 = E_Mahdi_ADC
	##################################### Layer4 ##################################################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L4 = E_SoTA
	energy_Mahdi_AlexNet_L4 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L4 = E_Mahdi_ADC
	##################################### Layer5 ##################################################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 256
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	E_Mahdi_ADC, T_Mahdi = energy_f.CL_i_bin_w_bin_Mahdi_ADC(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L5 = E_SoTA
	energy_Mahdi_AlexNet_L5 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L5 = E_Mahdi_ADC
	##################################### Layer6 ##################################################################
	
	
	input_channels = 9216
	output_channels = 4096
	datatype_size = 8
	
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA(input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi(input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC(input_channels, output_channels, datatype_size)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L6 = E_SoTA
	energy_Mahdi_AlexNet_L6 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L6 = E_Mahdi_ADC
	##################################### Layer7 ##################################################################
	
	
	input_channels = 4096
	output_channels = 4096
	datatype_size = 8
	
	
	E_SoTA,T_SoTA,SA_SoTA = energy_f.FC_i_bin_w_bin_SoTA(input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi,SA_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi(input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_bin_Mahdi_ADC(input_channels, output_channels, datatype_size)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	SA_activations_Mahdi = SA_activations_Mahdi + SA_Mahdi
	SA_activations_SoTA = SA_activations_SoTA + SA_SoTA
	
	energy_SoTA_AlexNet_L7 = E_SoTA
	energy_Mahdi_AlexNet_L7 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L7 = E_Mahdi_ADC
	##################################### Layer8 ##################################################################
	
	
	input_channels = 4096
	output_channels = 10
	datatype_size = 8
	
	
	E_SoTA,T_SoTA = energy_f.FC_i_bin_w_non( input_channels, output_channels, datatype_size)
	E_Mahdi, T_Mahdi = energy_f.FC_i_bin_w_non(input_channels, output_channels, datatype_size)
	E_Mahdi_ADC, T_Mahdi = energy_f.FC_i_bin_w_non(input_channels, output_channels, datatype_size)
	
	
	energy_SoTA = energy_SoTA + E_SoTA
	energy_Mahdi = energy_Mahdi + E_Mahdi
	energy_Mahdi_ADC = energy_Mahdi_ADC + E_Mahdi_ADC
	
	Transactions_SoTA = Transactions_SoTA + T_SoTA
	Transactions_Mahdi = Transactions_Mahdi + T_Mahdi
	
	
	energy_SoTA_AlexNet_L8 = E_SoTA
	energy_Mahdi_AlexNet_L8 = E_Mahdi 
	energy_Mahdi_ADC_AlexNet_L8 = E_Mahdi_ADC
	
	
	energy_SoTA_AlexNet = energy_SoTA 
	energy_Mahdi_AlexNet = energy_Mahdi
	energy_Mahdi_AlexNet_ADC = energy_Mahdi_ADC	
		
	Transaction_SoTA_AlexNet = Transactions_SoTA 
	Transaction_Mahdi_AlexNet = Transactions_Mahdi
	
	SA_activation_SoTA_AlexNet = SA_activations_SoTA 
	SA_activation_Mahdi_AlexNet = SA_activations_Mahdi
	

	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	#print('F2 SotA '+str(energy_SoTA_LeNet5/energy_Mahdi_LeNet5)+' '+str(energy_SoTA_MLP_S/energy_Mahdi_MLP_S)+' '+str(energy_SoTA_MLP_M/energy_Mahdi_MLP_M)+' '+str(energy_SoTA_MLP_L/energy_Mahdi_MLP_L)+' '+str(energy_SoTA_CNN1/energy_Mahdi_CNN1)+' '+str(energy_SoTA_CNN2/energy_Mahdi_CNN2)+' '+str(energy_SoTA_AlexNet/energy_Mahdi_AlexNet))
	#print('F2 SotA '+str(energy_Mahdi_ADC_LeNet5/energy_Mahdi_LeNet5)+' '+str(energy_Mahdi_MLP_S_ADC/energy_Mahdi_MLP_S)+' '+str(energy_Mahdi_MLP_M_ADC/energy_Mahdi_MLP_M)+' '+str(energy_Mahdi_MLP_L_ADC/energy_Mahdi_MLP_L)+' '+str(energy_Mahdi_CNN1_ADC/energy_Mahdi_CNN1)+' '+str(energy_Mahdi_CNN2_ADC/energy_Mahdi_CNN2)+' '+str(energy_Mahdi_AlexNet_ADC/energy_Mahdi_AlexNet))
	#print('F2 Mahdi '+str(energy_Mahdi_LeNet5)+''+str(energy_Mahdi_MLP_S)+''+str(energy_Mahdi_MLP_M)+''+str(energy_Mahdi_MLP_L)+''+str(energy_Mahdi_CNN1)+''+str(energy_Mahdi_CNN2)+''+str(energy_Mahdi_AlexNet))
	
	plt.rcParams.update({'font.size': 12})
	barWidth = 0.4
 
	F1=[energy_Mahdi_ADC_LeNet5,energy_Mahdi_MLP_S_ADC,energy_Mahdi_MLP_M_ADC,energy_Mahdi_MLP_L_ADC,energy_Mahdi_CNN1_ADC,energy_Mahdi_CNN2_ADC]
	F2=[energy_SoTA_LeNet5,energy_SoTA_MLP_S,energy_SoTA_MLP_M,energy_SoTA_MLP_L, energy_SoTA_CNN1, energy_SoTA_CNN2]
	F3=[energy_Mahdi_LeNet5,energy_Mahdi_MLP_S,energy_Mahdi_MLP_M,energy_Mahdi_MLP_L,energy_Mahdi_CNN1,energy_Mahdi_CNN2]
	F4=[0,0,0,0,energy_Mahdi_CNN1_SA3,energy_Mahdi_CNN2_SA3]
	
	r1 = np.arange(len(F1))*2
	fig, ax = plt.subplots()
	
	plt.bar(r1-2*barWidth, F1, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=1, label='Exact computing using ADC', hatch='//')	
	plt.bar(r1-1*barWidth, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=1, label='Baseline', hatch='**')
	plt.bar(r1+0*barWidth, F3, width = barWidth, capsize=1, color = 'honeydew', edgecolor = 'black', label='BCIM with 1 Ref', hatch='..')
	plt.bar(r1+1*barWidth, F4, width = barWidth, capsize=1, color = 'darkolivegreen', edgecolor = 'black', label='BCIM with 3 Ref', hatch='...')
	
	plt.xticks([r1*2 for r1 in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2'])
	
	ticks_y = tick.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e3))
	ax.yaxis.set_major_formatter(ticks_y)
	
	plt.ylabel('Energy (nJ)',fontsize=15)
	plt.tight_layout() 
	plt.legend()
	
	fig.set_size_inches(8, 3)
	plt.savefig("energy.pdf")
	
	#######################################################################################
	##########################AlexNET######################################################
	
	fig = plt.figure(figsize=(10, 3))
	coord1 = 121
	coord2 = 122
	fig.tight_layout() 
	gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
	######################
	fig1 = plt.subplot(gs[0])
	
	barWidth = 0.4
 
	F2=[energy_SoTA_AlexNet_L1, energy_SoTA_AlexNet_L2, energy_SoTA_AlexNet_L3, energy_SoTA_AlexNet_L4, energy_SoTA_AlexNet_L5, energy_SoTA_AlexNet_L6, energy_SoTA_AlexNet_L7, energy_SoTA_AlexNet_L8]
	F3=[energy_Mahdi_AlexNet_L1, energy_Mahdi_AlexNet_L2, energy_Mahdi_AlexNet_L3, energy_Mahdi_AlexNet_L4, energy_Mahdi_AlexNet_L5, energy_Mahdi_AlexNet_L6, energy_Mahdi_AlexNet_L7, energy_Mahdi_AlexNet_L8]
	F1=[energy_Mahdi_ADC_AlexNet_L1, energy_Mahdi_ADC_AlexNet_L2, energy_Mahdi_ADC_AlexNet_L3, energy_Mahdi_ADC_AlexNet_L4, energy_Mahdi_ADC_AlexNet_L5, energy_Mahdi_ADC_AlexNet_L6, energy_Mahdi_ADC_AlexNet_L7, energy_Mahdi_ADC_AlexNet_L8]
	
	r1 = 1.5*np.arange(len(F1))
	
	plt.bar(r1-barWidth, F1, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=1, label='Exact computing using ADC', hatch='//')	
	plt.bar(r1, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=1, label='Baseline', hatch='*')
	plt.bar(r1+barWidth, F3, width = barWidth, capsize=1, color = 'honeydew', edgecolor = 'black', label='BCIM', hatch='..')
	
	plt.xticks([r1*1.5 for r1 in range(len(F1))], ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'])
		
	ticks_y = tick.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e6))
	fig1.yaxis.set_major_formatter(ticks_y)

	plt.ylabel('Energy (uJ)',fontsize=15)		
	plt.legend()
		
	fig1 = plt.subplot(gs[1])
	
	F2=[energy_SoTA_AlexNet]
	F3=[energy_Mahdi_AlexNet]
	F1=[energy_Mahdi_AlexNet_ADC]
	r1 = np.arange(len(F1))
	
	barWidth = 0.3
	plt.bar(r1-barWidth, F1, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=1, label='Exact computing using ADC', hatch='//')
	
	plt.bar(r1, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=1, label='Baseline', hatch='*')
	plt.bar(r1+barWidth, F3, width = barWidth, capsize=1, color = 'honeydew', edgecolor = 'black', label='BCIM', hatch='..')
	
	plt.xticks([r for r in range(len(F1))], ['Total'])
	ticks_y = tick.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e6))
	fig1.yaxis.set_major_formatter(ticks_y)
	
	plt.savefig("energy_alexnet.pdf")
	
	#######################################################################################
	#######################################################################################
	
	plt.rcParams.update({'font.size': 14})
	barWidth = 0.3
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5]) 
	
	F1=[Transaction_Mahdi_LeNet5/Transaction_Mahdi_LeNet5,Transaction_Mahdi_CNN1/Transaction_Mahdi_CNN1,Transaction_Mahdi_CNN2/Transaction_Mahdi_CNN2, Transaction_Mahdi_AlexNet/Transaction_Mahdi_AlexNet]
	F2=[Transaction_SoTA_LeNet5/Transaction_Mahdi_LeNet5,Transaction_SoTA_CNN1/Transaction_Mahdi_CNN1,Transaction_SoTA_CNN2/Transaction_Mahdi_CNN2, Transaction_SoTA_AlexNet/Transaction_Mahdi_AlexNet]
	
	r1 = np.arange(len(F1))	
	fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
	
	ax1 = plt.subplot(gs[0])
	plt.bar(r1-barWidth, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=1, label='Baseline', hatch='*')
	plt.bar(r1, F1, width = barWidth, color = 'honeydew', edgecolor = 'black', capsize=1, label='BCIM', hatch='.')
		
	plt.xticks([r for r in range(len(F1))], ['LeNet5', 'CNN1', 'CNN2', 'AlexNet'])

	plt.ylabel('Relative transactions', fontsize=16)
	
	plt.xlabel('(a)')
	plt.legend()
	
	plt.legend(bbox_to_anchor=(0.2, 1.02, 1, 0.2), loc="lower left",
                borderaxespad=0, ncol=2)
	
	############################

	barWidth = 0.3
	ax1 = plt.subplot(gs[1])
	F1=[SA_activation_Mahdi_LeNet5/SA_activation_Mahdi_LeNet5,SA_activation_Mahdi_MLP_S/SA_activation_Mahdi_MLP_S,SA_activation_Mahdi_MLP_M/SA_activation_Mahdi_MLP_M,SA_activation_Mahdi_MLP_L/SA_activation_Mahdi_MLP_L,SA_activation_Mahdi_CNN1/SA_activation_Mahdi_CNN1,SA_activation_Mahdi_CNN2/SA_activation_Mahdi_CNN2, SA_activation_Mahdi_AlexNet/SA_activation_Mahdi_AlexNet]
	F2=[SA_activation_SoTA_LeNet5/SA_activation_Mahdi_LeNet5,SA_activation_SoTA_MLP_S/SA_activation_Mahdi_MLP_S,SA_activation_SoTA_MLP_M/SA_activation_Mahdi_MLP_M,SA_activation_SoTA_MLP_L/SA_activation_Mahdi_MLP_L,SA_activation_SoTA_CNN1/SA_activation_Mahdi_CNN1,SA_activation_SoTA_CNN2/SA_activation_Mahdi_CNN2, SA_activation_SoTA_AlexNet/SA_activation_Mahdi_AlexNet]
	
	r1 = np.arange(len(F1))
	
	plt.bar(r1, F1, width = barWidth, color = 'honeydew', edgecolor = 'black', capsize=1, label='BCIM', hatch='.')
	
	plt.bar(r1-barWidth, F2, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=1, label='Baseline', hatch='*')
	
	plt.xticks([r for r in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2', 'AlexNet'])

	plt.ylabel('Relative SA activations', fontsize=16)	
	plt.xlabel('(b)')
		
	plt.legend(bbox_to_anchor=(0.2, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)
	plt.savefig("SA_Activations_Transactions.pdf")
	
	######################################################################################
	######################################################################################
	######################################################################################
	######################################################################################
	
	fig, ax = plt.subplots()
	fig.set_figheight(3)
	fig.set_figwidth(6)
	colors1 = ( "steelblue","olive", "honeydew", "maroon", "teal")
	colors2 = ( "orange", "cyan", "brown","grey")
	hatches=  ['//', '**', '..', '/', '.']
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
	plt.tight_layout() 
	plt.savefig("energy_pie.pdf")
	
	#######################################################################################
	#######################################################################################
	
	
	
	