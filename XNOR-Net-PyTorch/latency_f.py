import math

#clock frequency for digital part is 1GHz
crossbar_row= 512
crossbar_columns= 512
digital_periphery_non_bin = 1

#in order to hide the latency of popcount, we instatiate a 4bit adder every 16 columns 
digital_periphery_bin =  1
data_communication_latency = 1 
 
#total read out would be around 100ns based on HD paper from IBM - All the numbers are in nano  
ADC_Latency = 3 #per signl conversion
SA_Latency = 1
SA3_Latency = 3
crossbar_Latency = []
latency_ADC_row = [] 
latency_SA_row = []
latency_SA3_row = []
shared_columns_SA = 4
shared_columns_ADC =  32
bus_width = 32

for i in range(crossbar_row):
	crossbar_Latency.append(i)

crossbar_Latency[2]=10
crossbar_Latency[511]=10
 


	
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

def CL_i_non_w_non_SoTA( kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name ):
	min_rows = (kernel_size_x * kernel_size_y)* input_channels
	
	temp_r = math.ceil(min_rows/crossbar_row)
	
	if(output_channels*datatype_size > crossbar_columns):
		temp_c = math.ceil(output_channels*datatype_size/crossbar_columns)
	else:
		temp_c = 1
				
	
	minimum_number_of_crossbars = temp_r*temp_c
	
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	
	###################### crossbars are operating in parallel #####################
	#calculating number of transaction per single output 
	number_of_transactions = ( input_channels *  kernel_size_y * kernel_size_x * 1) #input datatype size assumed to be one!
	
	############# latency per output ###############
	
	if(shared_columns_ADC >= datatype_size):
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ (crossbar_Latency[crossbar_row-1]+(ADC_Latency+digital_periphery_non_bin)*datatype_size)*1 #input datatype size assumed to be one!
	else:
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ (crossbar_Latency[crossbar_row-1]+(ADC_Latency+digital_periphery_non_bin)*shared_columns_ADC)*1 #input datatype size assumed to be one!
		
	############ Total latency #####################
	extra_transaction_per_line = input_channels *  kernel_size_y * kernel_size_x * 1 #input datatype size assumed to be one!
	Total_latency = latency_per_output * ((input_size_x-kernel_size_x)/stride+1)* ((input_size_y-kernel_size_y)/stride+1)
	
	#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))	
	
	return Total_latency


def CL_i_non_w_non_Mahdi(kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride, layer_name ):
	min_rows = (kernel_size_x * kernel_size_y)* input_channels
	
	temp_r = math.ceil(min_rows/crossbar_row)
	
	if(output_channels*datatype_size > crossbar_columns):
		temp_c = math.ceil(output_channels*datatype_size/crossbar_columns)
	else:
		temp_c = 1
				
	
	minimum_number_of_crossbars = temp_r*temp_c
	
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	
	###################### crossbars are operating in parallel #####################
	#calculating number of transaction per single output considering Mahdi's method 
	number_of_transactions = ( input_channels *  kernel_size_y * 1) * stride  #input datatype size assumed to be one!
	
	############# latency per output ###############
	
	if(shared_columns_ADC >= datatype_size):
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ (crossbar_Latency[crossbar_row-1]+(ADC_Latency+digital_periphery_non_bin)*datatype_size)*1 #input datatype size assumed to be one!
	else:
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ (crossbar_Latency[crossbar_row-1]+(ADC_Latency+digital_periphery_non_bin)*shared_columns_ADC)*1 #input datatype size assumed to be one!
		
	############ Total latency #####################
	extra_transaction_per_line = input_channels *  kernel_size_y * kernel_size_x * 1 #input datatype size assumed to be one!
	if(shared_columns_ADC >= datatype_size):
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+ADC_Latency*datatype_size)*1 #input datatype size assumed to be one!
	else:
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+ADC_Latency*shared_columns_ADC)*1 #input datatype size assumed to be one!
		
	Total_latency = (latency_per_output * (input_size_x-kernel_size_x)/stride + extra_latency)* ((input_size_y-kernel_size_y)/stride+1)
	
	#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))

	return Total_latency
      
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

   
def CL_i_bin_w_bin_Mahdi( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):

	min_rows = (kernel_size_x * kernel_size_y)* input_channels
	if( min_rows > crossbar_row ):
		temp_r = math.ceil(min_rows/crossbar_row)
		
		if(output_channels > crossbar_columns):
			temp_c = math.ceil(output_channels/crossbar_columns)
		else:
			temp_c = 1
					
		
		minimum_number_of_crossbars = temp_r*temp_c
		
		#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
		
		###################### crossbars are operating in parallel #####################
		#calculating number of transaction per single output considering Mahdi's method 
		number_of_transactions = ( input_channels *  kernel_size_y) * stride
	
		############# latency per output ###############
		
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ crossbar_Latency[crossbar_row-1]+ADC_Latency*shared_columns_SA
		
		############ Total latency #####################
		extra_transaction_per_line = input_channels *  kernel_size_y * kernel_size_x
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+ADC_Latency*shared_columns_SA)
		
		Total_latency = (latency_per_output * (input_size_x-kernel_size_x)/stride + extra_latency)* ((input_size_y-kernel_size_y)/stride+1)
		
		#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
		#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
		
	else: # min_rows < crossbar_row
		temp = crossbar_row - min_rows
		if(crossbar_columns >=output_channels):
			number_of_extra_outputs = min(math.floor(temp/(input_channels*kernel_size_y)), math.floor(crossbar_columns/output_channels))
		else:
			number_of_extra_outputs=0
		
		minimum_number_of_crossbars = math.ceil(output_channels/crossbar_columns)
		#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
		
		###################### crossbars are operating in parallel #####################
		#calculating number of transaction per single output considering Mahdi's method 
		number_of_transactions = ( input_channels *  kernel_size_y)*(stride + number_of_extra_outputs) 
	
		############# latency per output ###############
		
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ crossbar_Latency[crossbar_row-1]+ADC_Latency*shared_columns_SA
		
		############ Total latency #####################
		extra_transaction_per_line = input_channels *  kernel_size_y * (kernel_size_x+number_of_extra_outputs)
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+ADC_Latency*shared_columns_SA)
		
		Total_latency = (latency_per_output * (input_size_x-kernel_size_x)/(stride*(1+number_of_extra_outputs)) + extra_latency)* ((input_size_y-kernel_size_y)/stride+1)
		
		#print ('number of output generates per cycle for layer'+ layer_name + ' is:'+str((1+number_of_extra_outputs)))
		#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
		#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
		
	
	return Total_latency
   

def CL_i_bin_w_bin_Mahdi_SA3( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):

	min_rows = (kernel_size_x * kernel_size_y)* input_channels
	if( min_rows > crossbar_row ):
		temp_r = math.ceil(min_rows/crossbar_row)
		
		if(output_channels > crossbar_columns):
			temp_c = math.ceil(output_channels/crossbar_columns)
		else:
			temp_c = 1
					
		
		minimum_number_of_crossbars = temp_r*temp_c
		
		#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
		
		###################### crossbars are operating in parallel #####################
		#calculating number of transaction per single output considering Mahdi's method 
		number_of_transactions = ( input_channels *  kernel_size_y) * stride
	
		############# latency per output ###############
		
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ crossbar_Latency[crossbar_row-1]+SA3_Latency*shared_columns_SA
		
		############ Total latency #####################
		extra_transaction_per_line = input_channels *  kernel_size_y * kernel_size_x
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+SA3_Latency*shared_columns_SA)
		
		Total_latency = (latency_per_output * (input_size_x-kernel_size_x)/stride + extra_latency)* ((input_size_y-kernel_size_y)/stride+1)
		
		#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
		#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
		
	else: # min_rows < crossbar_row
		temp = crossbar_row - min_rows
		if(crossbar_columns >=output_channels):
			number_of_extra_outputs = min(math.floor(temp/(input_channels*kernel_size_y)), math.floor(crossbar_columns/output_channels))
		else:
			number_of_extra_outputs=0
		
		minimum_number_of_crossbars = math.ceil(output_channels/crossbar_columns)
		#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
		
		###################### crossbars are operating in parallel #####################
		#calculating number of transaction per single output considering Mahdi's method 
		number_of_transactions = ( input_channels *  kernel_size_y)*(stride + number_of_extra_outputs) 
	
		############# latency per output ###############
		
		latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ crossbar_Latency[crossbar_row-1]+SA3_Latency*shared_columns_SA
		
		############ Total latency #####################
		extra_transaction_per_line = input_channels *  kernel_size_y * (kernel_size_x+number_of_extra_outputs)
		extra_latency = data_communication_latency * (extra_transaction_per_line/bus_width)+ (crossbar_Latency[crossbar_row-1]+SA3_Latency*shared_columns_SA)
		
		Total_latency = (latency_per_output * (input_size_x-kernel_size_x)/(stride*(1+number_of_extra_outputs)) + extra_latency)* ((input_size_y-kernel_size_y)/stride+1)
		
		#print ('number of output generates per cycle for layer'+ layer_name + ' is:'+str((1+number_of_extra_outputs)))
		#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
		#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
		
	
	return Total_latency




   
def CL_i_bin_w_bin_SoTA( layer_name, kernel_size_x,kernel_size_y, input_channels, output_channels, datatype_size, input_size_x, input_size_y, stride ):
	
	min_columns = (kernel_size_x * kernel_size_y)* input_channels

	temp_r = math.ceil(min_columns/crossbar_columns)
	
	if(output_channels > crossbar_row):
		temp_c = math.ceil(output_channels/crossbar_row)
	else:
		temp_c = 1
				
	
	minimum_number_of_crossbars = temp_r*temp_c
	
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	
	###################### crossbars are operating in parallel #####################
	#calculating number of transaction per single output considering Mahdi's method 
	number_of_transactions = ( input_channels *  kernel_size_y * kernel_size_x) 
	
	############# latency per output ###############
	
	latency_per_output = data_communication_latency * (number_of_transactions/bus_width)+ (crossbar_Latency[2]+(SA_Latency+digital_periphery_bin)*shared_columns_SA)*output_channels
	
	############ Total latency #####################
	
	Total_latency = latency_per_output * ((input_size_x-kernel_size_x)/stride+1) * ((input_size_y-kernel_size_y)/stride+1)
	
	#print ('number of output generates per cycle for layer'+ layer_name + ' is: 1')
	#print ('latency_per_output for layer'+ layer_name + ' is:' + str(latency_per_output))
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
	
	return Total_latency
   
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################




##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################



def FC_i_bin_w_non( layer_name, input_channels, output_channels, datatype_size ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels
	temp_r = int(total_row_activations/crossbar_row +1)
	if(output_channels*datatype_size > crossbar_columns):
		temp_c = math.ceil(output_channels*datatype_size/crossbar_columns)
	else:
		temp_c = 1
						
	minimum_number_of_crossbars = temp_r*temp_c
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	

	#computation latency
	temp = ((crossbar_Latency[crossbar_row-1])+(ADC_Latency+ digital_periphery_non_bin)*shared_columns_ADC )
	
	#temp = ((energy_ADC_row[input_channels])*datatype_size + digital_periphery_non_bin) * output_channels
	number_of_transactions = input_channels
	
	Total_latency = temp + data_communication_latency * (number_of_transactions/bus_width)
	
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
	
	return Total_latency
   
   
def FC_i_bin_w_bin_Mahdi( layer_name, input_channels, output_channels ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels
	temp_r = int(total_row_activations/crossbar_row +1)
	if(output_channels > crossbar_columns):
		temp_c = math.ceil(output_channels/crossbar_columns)
	else:
		temp_c = 1
						
	minimum_number_of_crossbars = temp_r*temp_c
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	temp = ((crossbar_Latency[crossbar_row-1])+SA_Latency*shared_columns_SA)
	number_of_transactions = input_channels
	
	Total_latency = temp + data_communication_latency * (number_of_transactions/bus_width)
	
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
   
	return Total_latency


def FC_i_bin_w_bin_Mahdi_SA3( layer_name, input_channels, output_channels ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  input_channels
	temp_r = int(total_row_activations/crossbar_row +1)
	if(output_channels > crossbar_columns):
		temp_c = math.ceil(output_channels/crossbar_columns)
	else:
		temp_c = 1
						
	minimum_number_of_crossbars = temp_r*temp_c
	#print ('minimum number of crossbars for layer '+ layer_name+ 'is: ' + str(minimum_number_of_crossbars))
	temp = ((crossbar_Latency[crossbar_row-1])+ SA3_Latency*shared_columns_SA)
	number_of_transactions = input_channels
	
	Total_latency = temp + data_communication_latency * (number_of_transactions/bus_width)
	
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
   
	return Total_latency
	

def FC_i_bin_w_bin_SoTA( layer_name, input_channels, output_channels ):
	temp=0
	number_of_crossbars = 0
	total_row_activations = 0
	
	###########################################################################
	total_row_activations =  output_channels
	temp_r = int(total_row_activations/crossbar_row +1)
	if(input_channels > crossbar_columns):
		temp_c = math.ceil(output_channels/crossbar_columns)
	else:
		temp_c = 1
						
	minimum_number_of_crossbars = temp_r*temp_c
	#print ('minimum number of crossbars for layer '+ layer_name+ ' is: ' + str(minimum_number_of_crossbars))
	
	temp = ((crossbar_Latency[2])+(SA_Latency+ digital_periphery_bin)*shared_columns_SA)
	number_of_transactions = input_channels
	
	Total_latency = output_channels*temp + data_communication_latency * (number_of_transactions/bus_width)
	
	#print ('Total_latency for layer'+ layer_name + ' is:' + str(Total_latency))
   
	return Total_latency