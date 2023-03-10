import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import latency_f
import argparse
import matplotlib.ticker as tick
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
if __name__=='__main__':


	
	##################################### LeNet_5_new_S ##################################################################
	##################################### LeNet_5_new_S ##################################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0		
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
	
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
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)					   	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi= latency_f.CL_i_bin_w_bin_Mahdi(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)	
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi= latency_f.CL_i_bin_w_bin_Mahdi_ADC(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
		
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
		
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
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA  = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
		
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_LeNet_5_new_S_L2 = L_SoTA
	L_LeNet_5_new_S_L2_Mahdi = L_Mahdi
	
	################################## Layer3 #########################
	input_channels = 16*4*4
	output_channels = 120
	datatype_size = 8
	layer_name = 'LeNet5_Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)						 
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels)						 
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	
	L_LeNet_5_new_S_L3 = L_SoTA
	L_LeNet_5_new_S_L3_Mahdi = L_Mahdi
	################################## Layer4 #########################
       
	input_channels = 120
	output_channels = 84
	datatype_size = 8
	layer_name = 'LeNet5_Layer4'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels)
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_LeNet_5_new_S_L4 = L_SoTA
	L_LeNet_5_new_S_L4_Mahdi = L_Mahdi
	
	################################# Layer5 ############################
	input_channels = 84
	output_channels = 10
	datatype_size = 8
	layer_name = 'LeNet5_Layer5'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)					   
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	#####################################################################
	
	Latency_SoTA_LeNet5 = Latency_SoTA 
	Latency_Mahdi_LeNet5 = Latency_Mahdi
	Latency_Mahdi_LeNet5_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_LeNet5 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_LeNet5 = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_LeNet5_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_LeNet5 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_LeNet5 = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_LeNet5_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_LeNet5 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_LeNet5 = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_LeNet5_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	#######################
	L_LeNet_5_new_S_L5 = L_SoTA
	L_LeNet_5_new_S_L5_Mahdi = L_Mahdi
	
	
	##################################### CNN1 ##################################################################
	##################################### CNN1 ##################################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0

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
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)				   	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_ADC(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	

	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	layer_name = 'CNN1_Layer2'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)				 
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels)				 
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)

					   
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	layer_name = 'CNN1_Layer3'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels)					    	
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels)					    	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	
	layer_name = 'CNN1_Layer4'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)					   
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	#######################################################
	
	Latency_SoTA_CNN1 = Latency_SoTA 
	Latency_Mahdi_CNN1 = Latency_Mahdi
	Latency_Mahdi_CNN1_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_CNN1 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_CNN1 = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_CNN1_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_CNN1 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_CNN1 = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_CNN1_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_CNN1 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_CNN1 = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_CNN1_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	
	##################################### CNN1_SA3 ##################################################################
	##################################### CNN1_SA3 ##################################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	
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
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)					   
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
					   
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################## Layer2 #############################
	input_channels = 360
	output_channels = 720
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer2'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels)	                     
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
					   
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	################################## Layer3 #############################
       
	input_channels = 720
	output_channels = 70
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer3'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels)	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################## Layer4 #############################
	input_channels = 70
	output_channels = 10
	datatype_size = 8
	layer_name = 'CNN1_SA3_Layer4'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)					   
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	###############################################
	Latency_SoTA_CNN1_SA3 = Latency_SoTA 
	Latency_Mahdi_CNN1_SA3 = Latency_Mahdi
	
	Total_transaction_latency_SoTA_CNN1_SA3 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_CNN1_SA3 = Total_transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA_CNN1_SA3 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_CNN1_SA3 = Total_Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA_CNN1_SA3 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_CNN1_SA3 = Total_SA_and_periphery_latency_Mahdi
	
	##################################### CNN2 ##################################################################
	##################################### CNN2 ##################################################################	
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
	
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
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	#########
	L_CNN2_L1 = L_SoTA
	L_CNN2_L1_Mahdi = L_Mahdi
	
	
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_Layer2'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )	
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_CNN2_L2 = L_SoTA
	L_CNN2_L2_Mahdi = L_Mahdi
	
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_CNN2_L3 = L_SoTA
	L_CNN2_L3_Mahdi = L_Mahdi
	
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	
	layer_name = 'CNN2_Layer4'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	
	############################################
	
	Latency_SoTA_CNN2 = Latency_SoTA 
	Latency_Mahdi_CNN2 = Latency_Mahdi
	Latency_Mahdi_CNN2_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_CNN2 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_CNN2 = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_CNN2_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_CNN2 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_CNN2 = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_CNN2_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_CNN2 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_CNN2 = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_CNN2_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	L_CNN2_L4 = L_SoTA
	L_CNN2_L4_Mahdi = L_Mahdi
	##################################### CNN2_SA3 ##################################################################
	##################################### CNN2_SA3 ##################################################################	
	
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	
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
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride )

	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	################################# Layer2 ###################################
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	
	layer_name = 'CNN2_SA3.Layer2'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	################################# Layer3 ###################################
       
	input_channels = 1210
	output_channels = 1210
	datatype_size = 8
	layer_name = 'CNN2_SA3.Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################ Layer4 ####################################
	input_channels = 1210
	output_channels = 10
	datatype_size = 8
	layer_name = 'CNN2_SA3.Layer4'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size)
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	
	########################################
	Latency_SoTA_CNN2_SA3 = Latency_SoTA 
	Latency_Mahdi_CNN2_SA3 = Latency_Mahdi
	
	Total_transaction_latency_SoTA_CNN2_SA3 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_CNN2_SA3 = Total_transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA_CNN2_SA3 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_CNN2_SA3 = Total_Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA_CNN2_SA3 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_CNN2_SA3 = Total_SA_and_periphery_latency_Mahdi
	
	##################################### MLP-S ##################################################################
	##################################### MLP-S ##################################################################
		
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
	

	############################## Layer1 ###########################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-S_Layer1'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	############################## Layer2 ##########################
       
	input_channels = 784
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-S_Layer2'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	############################ Layer3 #############################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	layer_name = 'MLP-S_Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	######################### Layer4 #################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-S_Layer4'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	#########################################
	Latency_SoTA_MLP_S = Latency_SoTA 
	Latency_Mahdi_MLP_S = Latency_Mahdi
	Latency_Mahdi_MLP_S_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_MLP_S = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_MLP_S = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_MLP_S_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_MLP_S = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_MLP_S = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_MLP_S_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_MLP_S = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_MLP_S = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_MLP_S = Total_ADC_and_periphery_latency_Mahdi	
	
	########################################MLP-M######################################################################
	########################################MLP-M#####################################################################
	########################################MLP-M#####################################################################
		
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
		
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-M_Layer1'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1000
	datatype_size = 8
	layer_name = 'MLP-M_Layer2'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	##################################### Layer3 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-M_Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	##################################### Layer4 ##################################################################
       
	input_channels = 500
	output_channels = 250
	datatype_size = 8
	layer_name = 'MLP-M_Layer4'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	##################################### Layer5 ##################################################################
	input_channels = 250
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-M_Layer5'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	##################################
	
	Latency_SoTA_MLP_M = Latency_SoTA 
	Latency_Mahdi_MLP_M = Latency_Mahdi
	Latency_Mahdi_MLP_M_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_MLP_M = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_MLP_M = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_MLP_M_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_MLP_M = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_MLP_M = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_MLP_M_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_MLP_M = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_MLP_M = Total_SA_and_periphery_latency_Mahdi
	Total_SA_and_periphery_latency_Mahdi_MLP_M_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	
	##########################################'MLP-L'##############################################################
	##########################################'MLP-L'##############################################################
	##########################################'MLP-L'##############################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0	
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
	
	##################################### Layer1 ##################################################################
	input_channels = 28*28
	output_channels = 784
	datatype_size = 8
	layer_name = 'MLP-L_Layer1'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_MLP_L_L1_Mahdi = L_Mahdi
	##################################### Layer2 ##################################################################
       
	input_channels = 784
	output_channels = 1500
	datatype_size = 8
	layer_name = 'MLP-L_Layer2'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	L_MLP_L_L2_Mahdi = L_Mahdi
	##################################### Layer3 ##################################################################
       
	input_channels = 1500
	output_channels = 1000
	datatype_size = 8
	layer_name = 'MLP-L_Layer3'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	L_MLP_L_L3_Mahdi = L_Mahdi
	##################################### Layer4 ##################################################################
       
	input_channels = 1000
	output_channels = 500
	datatype_size = 8
	layer_name = 'MLP-L_Layer4'
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	L_MLP_L_L4_Mahdi = L_Mahdi
	##################################### Layer5 ##################################################################
	input_channels = 500
	output_channels = 10
	datatype_size = 8
	layer_name = 'MLP-L_Layer5'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	L_MLP_L_L5_Mahdi = L_Mahdi
	#################################################
	Latency_SoTA_MLP_L = Latency_SoTA 
	Latency_Mahdi_MLP_L = Latency_Mahdi
	Latency_Mahdi_MLP_L_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_MLP_L = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_MLP_L = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_MLP_L_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_MLP_L = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_MLP_L = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_MLP_L_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_MLP_L = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_MLP_L = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_MLP_L_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	########################################AlexNET##############################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	Latency_Mahdi_ADC = 0
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_transaction_latency_Mahdi_ADC = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_Crossbar_latency_Mahdi_ADC = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	Total_ADC_and_periphery_latency_Mahdi = 0
	############################### Layer1 #################################
	kernel_size_x = 11
	kernel_size_y = 11
	input_channels = 3
	output_channels = 96
	datatype_size = 8
	input_size_x = 256
	input_size_y = 256
	stride = 4
	layer_name = 'AlexNET_Layer1'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	###############################
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L1 = L_SoTA
	L_AlexNET_L1_Mahdi = L_Mahdi
	################################# Layer2 ###################################
	
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 96
	output_channels = 256
	datatype_size = 8
	input_size_x = 31
	input_size_y = 31
	stride = 1
	layer_name = 'AlexNET_Layer2'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#################################
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L2 = L_SoTA
	L_AlexNET_L2_Mahdi = L_Mahdi
	################################# Layer3 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 256
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer3'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L3 = L_SoTA
	L_AlexNET_L3_Mahdi = L_Mahdi
	################################# Layer4 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer4'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L4 = L_SoTA
	L_AlexNET_L4_Mahdi = L_Mahdi
	################################# Layer5 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 256
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer5'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.CL_i_bin_w_bin_Mahdi_ADC( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L5 = L_SoTA
	L_AlexNET_L5_Mahdi = L_Mahdi
	################################# Layer6 ###################################
	
	input_channels = 9216
	output_channels = 4096
	datatype_size = 8
	layer_name = 'AlexNET_Layer6'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L6 = L_SoTA
	L_AlexNET_L6_Mahdi = L_Mahdi
	################################# Layer7 ###################################
	
	input_channels = 4096
	output_channels = 4096
	datatype_size = 8
	layer_name = 'AlexNET_Layer7'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_bin_Mahdi_ADC( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L7 = L_SoTA
	L_AlexNET_L7_Mahdi = L_Mahdi
	##################################### Layer8 ##################################################################
	input_channels = 4096
	output_channels = 10
	datatype_size = 8
	layer_name = 'AlexNET_Layer8'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi_ADC, Transaction_latency_Mahdi_ADC, Crossbar_latency_Mahdi_ADC, ADC_and_periphery_latency_Mahdi =  latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	Latency_Mahdi_ADC = Latency_Mahdi_ADC + L_Mahdi_ADC
	Total_transaction_latency_Mahdi_ADC = Total_transaction_latency_Mahdi_ADC + Transaction_latency_Mahdi_ADC
	Total_Crossbar_latency_Mahdi_ADC = Total_Crossbar_latency_Mahdi_ADC + Crossbar_latency_Mahdi_ADC
	Total_ADC_and_periphery_latency_Mahdi = Total_ADC_and_periphery_latency_Mahdi + ADC_and_periphery_latency_Mahdi
	
	L_AlexNET_L8 = L_SoTA
	L_AlexNET_L8_Mahdi = L_Mahdi
	##########################
	Latency_SoTA_AlexNET = Latency_SoTA 
	Latency_Mahdi_AlexNET = Latency_Mahdi
	Latency_Mahdi_AlexNET_ADC = Latency_Mahdi_ADC
	
	Total_transaction_latency_SoTA_AlexNET = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_AlexNET = Total_transaction_latency_Mahdi
	Total_transaction_latency_Mahdi_AlexNET_ADC = Total_transaction_latency_Mahdi_ADC
	
	Total_Crossbar_latency_SoTA_AlexNET = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_AlexNET = Total_Crossbar_latency_Mahdi
	Total_Crossbar_latency_Mahdi_AlexNET_ADC = Total_Crossbar_latency_Mahdi_ADC
	
	Total_SA_and_periphery_latency_SoTA_AlexNET = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_AlexNET = Total_SA_and_periphery_latency_Mahdi
	Total_ADC_and_periphery_latency_Mahdi_AlexNET_ADC = Total_ADC_and_periphery_latency_Mahdi
	
	########################################AlexNET_SA3##############################################################
	########################################AlexNET_SA3##############################################################
	########################################AlexNET_SA3##############################################################
	########################################AlexNET_SA3##############################################################
	
	Latency_SoTA = 0
	Latency_Mahdi = 0
	
	Total_transaction_latency_SoTA = 0
	Total_transaction_latency_Mahdi = 0
	Total_Crossbar_latency_SoTA = 0
	Total_Crossbar_latency_Mahdi = 0
	Total_SA_and_periphery_latency_SoTA = 0
	Total_SA_and_periphery_latency_Mahdi = 0
	
	############################### Layer1 #################################
	kernel_size_x = 11
	kernel_size_y = 11
	input_channels = 3
	output_channels = 96
	datatype_size = 8
	input_size_x = 256
	input_size_y = 256
	stride = 4
	layer_name = 'AlexNET_Layer1'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi

	################################# Layer2 ###################################
	
	kernel_size_x = 5
	kernel_size_y = 5
	input_channels = 96
	output_channels = 256
	datatype_size = 8
	input_size_x = 31
	input_size_y = 31
	stride = 1
	layer_name = 'AlexNET_Layer2'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################# Layer3 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 256
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer3'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3(layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################# Layer4 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 384
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer4'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################# Layer5 ###################################
	
	kernel_size_x = 3
	kernel_size_y = 3
	input_channels = 384
	output_channels = 256
	datatype_size = 8
	input_size_x = 15
	input_size_y = 15
	stride = 1
	layer_name = 'AlexNET_Layer5'
	
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride)
	
	#print(E_SoTA)
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################# Layer6 ###################################
	
	input_channels = 9216
	output_channels = 4096
	datatype_size = 8
	layer_name = 'AlexNET_Layer6'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	################################# Layer7 ###################################
	
	input_channels = 4096
	output_channels = 4096
	datatype_size = 8
	layer_name = 'AlexNET_Layer7'
	
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels )
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	
	##################################### Layer8 ##################################################################
	input_channels = 4096
	output_channels = 10
	datatype_size = 8
	layer_name = 'AlexNET_Layer8'
	L_SoTA, Transaction_latency_SoTA, Crossbar_latency_SoTA, SA_and_periphery_latency_SoTA = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	L_Mahdi, Transaction_latency_Mahdi, Crossbar_latency_Mahdi, SA_and_periphery_latency_Mahdi = latency_f.FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size )
	
	Latency_SoTA = Latency_SoTA + L_SoTA
	Latency_Mahdi = Latency_Mahdi + L_Mahdi
	
	Total_transaction_latency_SoTA = Total_transaction_latency_SoTA + Transaction_latency_SoTA
	Total_transaction_latency_Mahdi = Total_transaction_latency_Mahdi + Transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA = Total_Crossbar_latency_SoTA + Crossbar_latency_SoTA
	Total_Crossbar_latency_Mahdi = Total_Crossbar_latency_Mahdi + Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA = Total_SA_and_periphery_latency_SoTA + SA_and_periphery_latency_SoTA
	Total_SA_and_periphery_latency_Mahdi = Total_SA_and_periphery_latency_Mahdi + SA_and_periphery_latency_Mahdi
	################################
	
	Latency_SoTA_AlexNET_SA3 = Latency_SoTA 
	Latency_Mahdi_AlexNET_SA3 = Latency_Mahdi
	
	Total_transaction_latency_SoTA_AlexNET_SA3 = Total_transaction_latency_SoTA 
	Total_transaction_latency_Mahdi_AlexNET_SA3 = Total_transaction_latency_Mahdi
	
	Total_Crossbar_latency_SoTA_AlexNET_SA3 = Total_Crossbar_latency_SoTA 
	Total_Crossbar_latency_Mahdi_AlexNET_SA3 = Total_Crossbar_latency_Mahdi
	
	Total_SA_and_periphery_latency_SoTA_AlexNET_SA3 = Total_SA_and_periphery_latency_SoTA 
	Total_SA_and_periphery_latency_Mahdi_AlexNET_SA3 = Total_SA_and_periphery_latency_Mahdi
	
	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	##############################################################################################################
	#print('F2 SotA '+str(Latency_Mahdi_LeNet5_ADC/Latency_Mahdi_LeNet5)+' '+str(Latency_Mahdi_MLP_S_ADC/Latency_Mahdi_MLP_S)+' '+str(Latency_Mahdi_MLP_M_ADC/Latency_Mahdi_MLP_M)+' '+str(Latency_Mahdi_MLP_L_ADC/Latency_Mahdi_MLP_L)+' '+str(Latency_Mahdi_CNN1_ADC/Latency_Mahdi_CNN2)+' '+str(Latency_Mahdi_CNN2_ADC/Latency_Mahdi_CNN2)+' '+str(Latency_Mahdi_AlexNET_ADC/Latency_Mahdi_AlexNET))
	
	
	######################
	barWidth = 0.4
 
	F1=[Latency_SoTA_LeNet5/Latency_SoTA_LeNet5,Latency_SoTA_MLP_S/Latency_SoTA_MLP_S,Latency_SoTA_MLP_M/Latency_SoTA_MLP_M,Latency_SoTA_MLP_L/Latency_SoTA_MLP_L, Latency_SoTA_CNN1/Latency_SoTA_CNN1, Latency_SoTA_CNN2/Latency_SoTA_CNN2, Latency_SoTA_AlexNET/Latency_SoTA_AlexNET]
	F2=[Latency_SoTA_LeNet5/Latency_Mahdi_LeNet5_ADC,Latency_SoTA_MLP_S/Latency_Mahdi_MLP_S_ADC,Latency_SoTA_MLP_M/Latency_Mahdi_MLP_M_ADC,Latency_SoTA_MLP_L/Latency_Mahdi_MLP_L_ADC,Latency_SoTA_CNN1/Latency_Mahdi_CNN1_ADC,Latency_SoTA_CNN2/Latency_Mahdi_CNN2_ADC, Latency_SoTA_AlexNET/Latency_Mahdi_AlexNET_ADC]
	F3=[Latency_SoTA_LeNet5/Latency_Mahdi_LeNet5,Latency_SoTA_MLP_S/Latency_Mahdi_MLP_S,Latency_SoTA_MLP_M/Latency_Mahdi_MLP_M,Latency_SoTA_MLP_L/Latency_Mahdi_MLP_L,Latency_SoTA_CNN1/Latency_Mahdi_CNN1,Latency_SoTA_CNN2/Latency_Mahdi_CNN2, Latency_SoTA_AlexNET/Latency_Mahdi_AlexNET]
	F4=[0,0,0,0,Latency_SoTA_CNN1/Latency_Mahdi_CNN1_SA3,Latency_SoTA_CNN2/Latency_Mahdi_CNN2_SA3, Latency_SoTA_AlexNET/Latency_Mahdi_AlexNET_SA3]
	
	r1 = 2*np.arange(len(F1))
	fig, ax = plt.subplots()
	
	plt.bar(r1-2*barWidth, F1, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=7, label='Baseline', hatch='\\') 
	plt.bar(r1-barWidth, F2, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=7, label='Exact computing using ADC', hatch='//')
	plt.bar(r1, F3, width = barWidth, color = 'honeydew', edgecolor = 'black', capsize=7, label='BCIM with 1 Ref', hatch='*')
	plt.bar(r1+barWidth, F4, width = barWidth, capsize=7, color = 'darkolivegreen', edgecolor = 'black', label='BCIM with 3 Ref', hatch='.')
	
	plt.xticks([r*2 for r in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2', 'AlexNET'])
	
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:}x'.format(x) for x in vals])
	plt.ylabel('Latency improvement',fontsize=14)
	plt.xticks(fontsize=14, rotation=45)
	plt.tight_layout() 
	plt.legend()
	plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
	fig.set_size_inches(9, 4)
	
	fig.tight_layout() 
	ax.set_yscale('log')
	plt.savefig("Latency.pdf")
	
	
	######################################################################################
	######################################################################################
	barWidth = 0.4
	
	fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
	
	F1=[Total_transaction_latency_SoTA_LeNet5/Latency_SoTA_LeNet5, Total_transaction_latency_SoTA_MLP_S/Latency_SoTA_MLP_S, Total_transaction_latency_SoTA_MLP_M/Latency_SoTA_MLP_M, Total_transaction_latency_SoTA_MLP_L/Latency_SoTA_MLP_L, Total_transaction_latency_SoTA_CNN1/Latency_SoTA_CNN1, Total_transaction_latency_SoTA_CNN2/Latency_SoTA_CNN2, Total_transaction_latency_SoTA_AlexNET/Latency_SoTA_AlexNET]
	F2=[Total_Crossbar_latency_SoTA_LeNet5/Latency_SoTA_LeNet5, Total_Crossbar_latency_SoTA_MLP_S/Latency_SoTA_MLP_S, Total_Crossbar_latency_SoTA_MLP_M/Latency_SoTA_MLP_M, Total_Crossbar_latency_SoTA_MLP_L/Latency_SoTA_MLP_L, Total_Crossbar_latency_SoTA_CNN1 /Latency_SoTA_CNN1, Total_Crossbar_latency_SoTA_CNN2 /Latency_SoTA_CNN2, Total_Crossbar_latency_SoTA_AlexNET /Latency_SoTA_AlexNET]
	F3=[Total_SA_and_periphery_latency_SoTA_LeNet5/Latency_SoTA_LeNet5, Total_SA_and_periphery_latency_SoTA_MLP_S/Latency_SoTA_MLP_S, Total_SA_and_periphery_latency_SoTA_MLP_M/Latency_SoTA_MLP_M, Total_SA_and_periphery_latency_SoTA_MLP_L/Latency_SoTA_MLP_L, Total_SA_and_periphery_latency_SoTA_CNN1/Latency_SoTA_CNN1, Total_SA_and_periphery_latency_SoTA_CNN2/Latency_SoTA_CNN2, Total_SA_and_periphery_latency_SoTA_AlexNET/Latency_SoTA_AlexNET]
	
	F4=[Total_transaction_latency_Mahdi_LeNet5/Latency_Mahdi_LeNet5, Total_transaction_latency_Mahdi_MLP_S/Latency_Mahdi_MLP_S, Total_transaction_latency_Mahdi_MLP_M/Latency_Mahdi_MLP_M, Total_transaction_latency_Mahdi_MLP_L/Latency_Mahdi_MLP_L, Total_transaction_latency_Mahdi_CNN1/Latency_Mahdi_CNN1, Total_transaction_latency_Mahdi_CNN2/Latency_Mahdi_CNN2, Total_transaction_latency_Mahdi_AlexNET/Latency_Mahdi_AlexNET]
	F5=[Total_Crossbar_latency_Mahdi_LeNet5/Latency_Mahdi_LeNet5, Total_Crossbar_latency_Mahdi_MLP_S/Latency_Mahdi_MLP_S, Total_Crossbar_latency_Mahdi_MLP_M/Latency_Mahdi_MLP_M, Total_Crossbar_latency_Mahdi_MLP_L/Latency_Mahdi_MLP_L, Total_Crossbar_latency_Mahdi_CNN1 /Latency_Mahdi_CNN1, Total_Crossbar_latency_Mahdi_CNN2 /Latency_Mahdi_CNN2, Total_Crossbar_latency_Mahdi_AlexNET /Latency_Mahdi_AlexNET]
	F6=[Total_SA_and_periphery_latency_Mahdi_LeNet5/Latency_Mahdi_LeNet5, Total_SA_and_periphery_latency_Mahdi_MLP_S/Latency_Mahdi_MLP_S, Total_SA_and_periphery_latency_Mahdi_MLP_M/Latency_Mahdi_MLP_M, Total_SA_and_periphery_latency_Mahdi_MLP_L/Latency_Mahdi_MLP_L, Total_SA_and_periphery_latency_Mahdi_CNN1/Latency_Mahdi_CNN1, Total_SA_and_periphery_latency_Mahdi_CNN2/Latency_Mahdi_CNN2, Total_SA_and_periphery_latency_Mahdi_AlexNET/Latency_Mahdi_AlexNET]
	
	r1 = 2*np.arange(len(F1))
	fig, ax = plt.subplots()
	
	ax = plt.subplot(2, 1, 1)
	plt.bar(r1-2*barWidth, F1, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=7, label='Transaction_latency', hatch='\\') 
	plt.bar(r1-barWidth, F2, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=7, label='Crossbar_latency', hatch='..')
	plt.bar(r1, F3, width = barWidth, color = 'honeydew', edgecolor = 'black', capsize=7, label='Periphery_latency', hatch='*')
	#plt.xticks([r*2 for r in range(len(F1))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2', 'AlexNET'])
	plt.xticks([])
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
	plt.legend()
	plt.legend(bbox_to_anchor=(0.1, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=3)
	plt.xlabel('(a) Baseline',fontsize=14)
	
	
	ax = plt.subplot(2, 1, 2)
	plt.bar(r1-2*barWidth, F4, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=7, label='Transaction_latency', hatch='\\') 
	plt.bar(r1-barWidth, F5, width = barWidth, color = 'steelblue', edgecolor = 'black', capsize=7, label='Crossbar_latency', hatch='..')
	plt.bar(r1, F6, width = barWidth, color = 'honeydew', edgecolor = 'black', capsize=7, label='Periphery_latency', hatch='*')
	
	plt.rcParams.update({'font.size': 14})
	plt.xticks([r*2 for r in range(len(F4))], ['LeNet5', 'MLP-S', 'MLP-M', 'MLP-L', 'CNN1', 'CNN2', 'AlexNET'])
	
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
	plt.xlabel('(b) BCIM',fontsize=14)
	plt.xticks(fontsize=14, rotation=45)
	fig.text(0.00, 0.51, 'Contribution to the latency', va='center', rotation='vertical',fontsize=14)
	plt.tight_layout() 
	
	fig.set_size_inches(9.5, 6.5)
	fig.tight_layout() 
	
	plt.savefig("Latency_breakdown.pdf")
	#######################################################################################
	#######################################################################################
	
	colors1 = ( "silver","navy", "olive", "maroon", "teal")
	colors2 = ( "silver","navy", "olive", "maroon", "teal", "orange", "cyan", "brown","grey")
	
	fig = plt.figure()
	fig.set_figheight(4)
	fig.set_figwidth(8)
	
	ax = plt.subplot(1, 2, 1)
	y2 = np.array([L_AlexNET_L1_Mahdi,L_AlexNET_L2_Mahdi,L_AlexNET_L3_Mahdi,L_AlexNET_L4_Mahdi,L_AlexNET_L5_Mahdi,L_AlexNET_L6_Mahdi,L_AlexNET_L7_Mahdi,L_AlexNET_L8_Mahdi])
	mylabels2 = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "Layer7", "Layer8"]
	myexplode2 = [0.1, 0, 0, 0, 0, 0, 0, 0]	
	patches, texts = plt.pie(y2, explode = myexplode2, colors=colors2)
	plt.legend(patches, mylabels2, loc="best")
	ax.set_title("AlexNet latency per layer")
		
	ax = plt.subplot(1, 2, 2)	
	y2 = np.array([L_MLP_L_L1_Mahdi, L_MLP_L_L2_Mahdi, L_MLP_L_L3_Mahdi, L_MLP_L_L4_Mahdi, L_MLP_L_L5_Mahdi])
	mylabels2 = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]  
	myexplode2 = [0.1, 0, 0, 0, 0]	
	ax.set_title("MLP-L latency per layer")
	
	patches, texts = plt.pie(y2, explode = myexplode2, colors=colors1)
	plt.legend(patches, mylabels2, loc="best")
		
	plt.tight_layout() 
	plt.savefig("Latency_CNN2_AlexNet.pdf")
	
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
	plt.savefig("Latency_pie_LeNet5.pdf")
	
	