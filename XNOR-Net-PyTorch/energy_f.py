
crossbar_row= 512
crossbar_columns= 512
# numbers are pJ

# energy = 7uW*1ns 
digital_periphery_non_bin = 0.007
# energy = 3uW*1ns
digital_periphery_bin = 0.003

# energy = 5mW*1ns
data_communication_energy = 5
# energy ~ 2.6mW*3ns
ADC_energy = 12 # last page of "In-memory hyperdimensional computing" paper from Manuel Le Gallo

SA_energy = 0.01
SA3_energy = 0.01*3
crossbar_energy = []
energy_ADC_row = []
energy_SA_row = []
energy_SA3_row = []


for i in range(crossbar_row):
	crossbar_energy.append(10*i/crossbar_row)
	
#energy per columns 	
crossbar_energy[511]=10 #1280   # Average energy - Assuming LRS=5k HRS=1G and Vread=0.2 - Half of the devices are in LRS and Half HRS. Half of the worlines are active at the same time due to the complemenatry wordlines 
crossbar_energy[2]=0.04




for i in range(crossbar_row):
	energy_ADC_row.append(crossbar_energy[i] + ADC_energy) 
	energy_SA_row.append(crossbar_energy[i] + SA_energy) 
	energy_SA3_row.append(crossbar_energy[i] + SA3_energy)

bus_width = 32 

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

def CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ): #input datatype size assumed to be one!
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################
	total_row_activations =  (kernel_size_x*kernel_size_y)*input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	###########################
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):
			temp = temp + ((energy_ADC_row[total_row_activations % crossbar_row-1]+digital_periphery_non_bin)* datatype_size)* 1 * output_channels #input datatype size assumed to be one!
		else:			
			temp = temp + ((energy_ADC_row[crossbar_row-1]+digital_periphery_non_bin)* datatype_size) * 1 * output_channels #input datatype size assumed to be one!
		
	number_of_transactions = input_channels*(kernel_size_x*kernel_size_y)* 1 * ((input_size_x-kernel_size_x)/stride +1 ) * ((input_size_y-kernel_size_y)/stride + 1) #input datatype size assumed to be one!
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
		
	return energy, number_of_transactions/bus_width


def CL_i_non_w_non_Mahdi( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ): #input datatype size assumed to be one!
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	#############################
	total_row_activations =  (kernel_size_x*kernel_size_y)*input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	#############################
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):
			temp = temp + ((energy_ADC_row[total_row_activations % crossbar_row-1]+digital_periphery_non_bin)* datatype_size)* 1 * output_channels #input datatype size assumed to be one!
		else:			
			temp = temp + ((energy_ADC_row[crossbar_row-1]+digital_periphery_non_bin)* datatype_size) * 1 * output_channels #input datatype size assumed to be one!
		
	number_of_transactions = ( input_channels * 1 * ((input_size_y-kernel_size_y)/stride+1)) * ((kernel_size_x*kernel_size_y)+ ((input_size_x-kernel_size_x))* kernel_size_y) #input datatype size assumed to be one!
	
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
	
	return energy, number_of_transactions/bus_width
      
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

   
def CL_i_bin_w_bin_Mahdi( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):

	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  2*(kernel_size_x*kernel_size_y)*input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	###########################################################################
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):
			temp = temp + (energy_SA_row[total_row_activations % crossbar_row-1]) * output_channels
		else:			
			temp = temp + (energy_SA_row[crossbar_row-1]) * output_channels
	
	
	number_of_transactions = ( input_channels * ((input_size_y-kernel_size_y)/stride +1 )) * ((kernel_size_x*kernel_size_y)+ ((input_size_x-kernel_size_x))* kernel_size_y)
	
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
	
	Number_of_SA_activations = ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride)
	return energy, number_of_transactions/bus_width, Number_of_SA_activations
	
def CL_i_bin_w_bin_Mahdi_ADC( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):

	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  2*(kernel_size_x*kernel_size_y)*input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	###########################################################################
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):
			temp = temp + (energy_ADC_row[total_row_activations % crossbar_row-1]) * output_channels
		else:			
			temp = temp + (energy_ADC_row[crossbar_row-1]) * output_channels
	
	
	number_of_transactions = ( input_channels * ((input_size_y-kernel_size_y)/stride +1 )) * ((kernel_size_x*kernel_size_y)+ ((input_size_x-kernel_size_x))* kernel_size_y)
	
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
	
	return energy, number_of_transactions/bus_width

def CL_i_bin_w_bin_Mahdi_SA3( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):

	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  2*(kernel_size_x*kernel_size_y)*input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	###########################################################################
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):
			temp = temp + (energy_SA3_row[total_row_activations % crossbar_row-1]) * output_channels
		else:			
			temp = temp + (energy_SA3_row[crossbar_row-1]) * output_channels
	
	
	number_of_transactions = ( input_channels * ((input_size_y-kernel_size_y)/stride +1 )) * ((kernel_size_x*kernel_size_y)+ ((input_size_x-kernel_size_x))* kernel_size_y)
	
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
	
	Number_of_SA_activations = ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride)
	
	return energy, number_of_transactions/bus_width, Number_of_SA_activations   
   
def CL_i_bin_w_bin_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  output_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	###########################################################################
	
	temp = (((energy_SA_row[2]+digital_periphery_bin) * (kernel_size_x*kernel_size_y)*input_channels))*total_row_activations
	
	number_of_transactions = input_channels*(kernel_size_x*kernel_size_y) * ((input_size_x-kernel_size_x)/stride+1) * ((input_size_y-kernel_size_y)/stride+1)
	
	energy = temp * ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) + data_communication_energy * (number_of_transactions/bus_width)
	
	Number_of_SA_activations = ((input_size_x-kernel_size_x)/stride) * ((input_size_y-kernel_size_y)/stride) * total_row_activations
	return energy, number_of_transactions/bus_width, Number_of_SA_activations
   
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################




##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################



def FC_i_bin_w_non( input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):			
			temp = temp + ((energy_ADC_row[total_row_activations % crossbar_row-1]+ digital_periphery_non_bin)*datatype_size) * output_channels
		else:						
			temp = temp + ((energy_ADC_row[crossbar_row-1]+ digital_periphery_non_bin)*datatype_size + digital_periphery_non_bin) * output_channels
	
	number_of_transactions = input_channels	
	energy = temp + data_communication_energy * (number_of_transactions/bus_width)
	
	return energy, number_of_transactions/bus_width
   
   
def FC_i_bin_w_bin_Mahdi( input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels * 2
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):			
			temp = temp + (energy_SA_row[total_row_activations % crossbar_row -1])*output_channels
		else:						
			temp = temp + (energy_SA_row[crossbar_row-1])*output_channels
	
	number_of_transactions = input_channels
	
	energy = temp + data_communication_energy * (number_of_transactions/bus_width)
	Number_of_SA_activations = 1   
	
	return energy, number_of_transactions/bus_width, Number_of_SA_activations

def FC_i_bin_w_bin_Mahdi_ADC( input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels * 2
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):			
			temp = temp + (energy_ADC_row[total_row_activations % crossbar_row -1])*output_channels
		else:						
			temp = temp + (energy_ADC_row[crossbar_row-1])*output_channels
	

	number_of_transactions = input_channels
	
	energy = temp + data_communication_energy * (number_of_transactions/bus_width)
	   
	return energy, number_of_transactions/bus_width	
   

def FC_i_bin_w_bin_Mahdi_SA3( input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels * 2
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	
	for x in range(number_of_crossbars):
		if(x==number_of_crossbars-1):			
			#temp = temp + (energy_SA3_row[total_row_activations % crossbar_row])*output_channels
			temp = temp + (energy_SA3_row[total_row_activations % crossbar_row-1])*output_channels
		else:						
			temp = temp + (energy_SA3_row[crossbar_row-1])*output_channels
	
	
	#temp = (energy_SA_row[total_row_activations])*output_channels
	number_of_transactions = input_channels
	
	energy = temp + data_communication_energy * (number_of_transactions/bus_width)
	   
   
	Number_of_SA_activations = 1
	return energy, number_of_transactions/bus_width, Number_of_SA_activations 
	
	
def FC_i_bin_w_bin_SoTA( input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	#print('SoTA_FC')
	###########################################################################
	total_row_activations =  output_channels
	number_of_crossbars = int(total_row_activations/crossbar_row +1)
	
	temp = (((energy_SA_row[2]+ digital_periphery_bin)*input_channels ))*total_row_activations
	
	number_of_transactions = input_channels
	
	energy = temp + data_communication_energy * (number_of_transactions/bus_width)
	Number_of_SA_activations = total_row_activations
	return energy, number_of_transactions/bus_width, Number_of_SA_activations