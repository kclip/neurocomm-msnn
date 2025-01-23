
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Set environment variable to restrict TensorFlow to GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import sionna as sn
import numpy as np
import scipy

# # List all available GPUs
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         # Restrict TensorFlow to use only the first GPU (GPU 0)
#         tf.config.set_visible_devices(gpus[0], 'GPU')
        
#         # Optionally, set memory growth to avoid TensorFlow pre-allocating all GPU memory
#         tf.config.experimental.set_memory_growth(gpus[0], True)
        
#         # print("Using only GPU 0")
#     except RuntimeError as e:
#         print(e)

class Conv2dLayer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.0,bias=True,batch_norm=True):
        super(Conv2dLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x



class LinearLayer(nn.Module):
    def __init__(self,in_plane,out_plane,droprate=0.0,bias=True,batch_norm=True):
        super(LinearLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane),
                nn.BatchNorm1d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x



class VGG_Gesture(nn.Module):
    def __init__(self,output_dim = 11):
        super(VGG_Gesture, self).__init__()
        pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
            Conv2dLayer(64,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            nn.Flatten(1,-1)
        )
        W = int(128/4/2/2/2/2/2)
        self.fc =  LinearLayer(128*W*W,512,droprate=0.0)
        self.classifier = nn.Linear(512,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class VGG_Decoder(nn.Module):
    def __init__(self,output_dim = 11):
        super(VGG_Gesture, self).__init__()
        pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
            Conv2dLayer(64,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            nn.Flatten(1,-1)
        )
        W = int(128/4/2/2/2/2/2)
        self.fc =  LinearLayer(128*W*W,512,droprate=0.0)
        self.classifier = nn.Linear(512,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        x = self.classifier(x)
        return x

# class wirelss_ch(nn.Module):
#     def __init__(self, N_t,num_steps, output_dim = 11):
#         super(wirelss_ch, self).__init__()  
#         self.N_t = N_t
#         self.batch_size = N_t
#         self.num_steps = num_steps
#         self.guard = 5 # guard period for avoiding the inter-symbol interference due to delay spread
#         self.N_c = 3  # number of delayed version of the signal

#     def forward(self, input):
#         batch_size = input.size()[1]
#         N_t = self.N_t
#         N_c = self.N_c
#         num_steps = self.num_steps
#         num_bw_expansion = N_t
#         guard = self.guard
#         spk_rec, mem_rec = input  # [num_steps, batch_size, num_output]
#         spk_rec = torch.transpose(spk_rec, 0, 1)  # [batch_size, num_steps,  num_output]
#         data_modulation = torch.zeros(N_t, batch_size, num_steps, num_bw_expansion + guard)
#         for n in range(N_t):  # For each antenna's data stream
#             data_modulation[n, :, :, n] = spk_rec[:, :, n]  # symbol 1 is modulated to position L_b

#         shape_data = torch.Tensor.size(data_modulation[0])  # [batch_size, num_steps, num_bw_expansion + guard]
#         shape_noise = shape_data

#         # First define the delayed version of the signal from each antenna
#         data_delay = torch.zeros(N_t, N_c + 1, batch_size, num_steps, num_bw_expansion + guard)
#         for n in range(N_t):  # the nth transmit antenna
#             for j in range(N_c + 1):  # (j+1) delay for signal from the nth transmit antenna to rth receive antenna
#                 if j == 0:  # 1 delay
#                     data_delay[n, j] = torch.roll(data_modulation[n], 1)  # 1 delay for the signal from the nth antenna
#                     data_delay[n, j, :, 0, 0] = 0
#                 if j == 1:  # 2 delay
#                     data_delay[n, j] = torch.roll(data_modulation[n], 2)  # 2 delay for the signal from the nth antenna
#                     data_delay[n, j, :, 0, 0] = 0
#                     data_delay[n, j, :, 0, 1] = 0
#                 if j == 2:  # 3 delay
#                     data_delay[n, j] = torch.roll(data_modulation[n], 3)  # 3 delay for the nth to rth antenna
#                     data_delay[n, j, :, 0, 0] = 0
#                     data_delay[n, j, :, 0, 1] = 0
#                     if torch.Tensor.size(data_modulation[n])[2] > 2:
#                         data_delay[n, j, :, 0, 2] = 0
#                     else:
#                         data_delay[n, j, :, 1, 0] = 0
#                 if j == 3:  # no delay
#                     data_delay[n, j] = data_modulation[n]
#         data_delay = data_delay + 1j * 0
#         channel_fix = 0

#         if channel_fix:
#             channel = torch.zeros(N_t, N_c + 1, batch_size, num_steps, num_bw_expansion + guard) + 1j * 0
#             for n in range(N_t):  # the nth transmit antenna
#                 for j in range(N_c + 1):  # (j+1) delay for signal from the nth transmit antenna
#                     channel_batch = 0.1 * torch.ones(batch_size, 1, 1) + 1j * 0.1 * torch.ones(batch_size, 1, 1)
#                     channel[n, j] = channel_batch.repeat(1, num_steps, num_bw_expansion + guard)
#         else:
#             channel = torch.zeros(N_t, N_c + 1, batch_size, num_steps, num_bw_expansion + guard) + 1j * 0
#             for n in range(N_t):  # the nth transmit antenna
#                 for j in range(N_c + 1):  # (j+1) delay for signal from the nth transmit antenna
#                     channel_batch = torch.normal(mean=0, std=np.sqrt(1/2), size=(batch_size, 1, 1)) + 1j * torch.normal(mean=0, std=np.sqrt(1/2), size=(batch_size, 1, 1))
#                     channel[n, j] = channel_batch.repeat(1, num_steps, num_bw_expansion + guard)

#         decoder_input = torch.zeros(batch_size, num_steps, num_bw_expansion + guard) + 1j * 0
#         for n in range(N_t):  # the nth transmit antenna
#             for j in range(N_c + 1):  # (j+1) delay for signal from the nth transmit antenna
#                 decoder_input = decoder_input + data_delay[n, j] * channel[n, j]

#         decoder_input = decoder_input + torch.normal(mean=0, std=np.sqrt(1 / 50), size=shape_noise)
#         decoder_input = torch.transpose(decoder_input, 0, 1)  # [num_steps, batch_size, num_bw_expansion + guard]
#         decoder_input = torch.cat((torch.real(decoder_input), torch.imag(decoder_input)), 2)  

#         # Hypernetwork Module
#         if enable_hyper == 1:
#             # 1) pilot transmission:
#             pilot_modulated = torch.zeros(N_t, batch_size, num_bw_expansion + guard)
#             for n in range(N_t):  # For each antenna's data stream
#                 pilot_modulated[n, :, n] = torch.ones(batch_size)  # symbol 0 is modulated to position 0
#             # 2) delayed version of the pilots:
#             pilot_delay = torch.zeros(N_t, N_c + 1, batch_size, num_bw_expansion + guard)
#             for n in range(N_t):  # the nth transmit antenna
#                 for j in range(N_c + 1):  # (j+1) delay for signal from the nth transmit antenna to rth receive antenna
#                     if j == 0:  # 1 delay
#                         pilot_delay[n, j] = torch.roll(pilot_modulated[n], 1)  # 1 delay for the nth to rth antenna
#                         pilot_delay[n, j, :, 0] = 0
#                     if j == 1:  # 2 delay
#                         pilot_delay[n, j] = torch.roll(pilot_modulated[n], 2)  # 2 delay for the nth to rth antenna
#                         pilot_delay[n, j, :, 0] = 0
#                         pilot_delay[n, j, :, 1] = 0
#                     if j == 2:  # 3 delay
#                         pilot_delay[n, j] = torch.roll(pilot_modulated[n], 3)  # 3 delay for the nth to rth antenna
#                         pilot_delay[n, j, :, 0] = 0
#                         pilot_delay[n, j, :, 1] = 0
#                         if torch.Tensor.size(pilot_modulated[n])[1] > 2:
#                             pilot_delay[n, j, :, 2] = 0
#                         else:
#                             pilot_delay[n, j, 1, 0] = 0
#                     if j == 3:  # no delay
#                         pilot_delay[n, j] = pilot_modulated[n]
#             pilot_delay = pilot_delay + 1j * 0
#             # 3) pass the PPM-modulated pilots over the channel:
#             input_hypernetwork = torch.zeros(batch_size, num_bw_expansion + guard) + 1j * 0
#             for n in range(N_t):
#                 for j in range(N_c + 1):
#                     input_hypernetwork = input_hypernetwork + pilot_delay[n, j] * channel[n, j, :, 0, :]
#             input_hypernetwork = input_hypernetwork + torch.normal(mean=0, std=np.sqrt(1 / 50), size=(batch_size, num_bw_expansion + guard))
#             input_hyper = torch.cat((torch.real(input_hypernetwork), torch.imag(input_hypernetwork)), 1)
#             # 4) pass the received pilots to the hypernetwork:
#             hypernetwork.train()
#             output_hyper = hypernetwork(input_hyper)  # [batch_size, num_inputs_de + num_hidden_de]
            
#         return x




class VGG_Encoder(nn.Module):
    def __init__(self,output_dim = 11):
        super(VGG_Encoder, self).__init__()
        pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
            Conv2dLayer(64,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            nn.Flatten(1,-1)
        )
        W = int(128/4/2/2/2/2/2)
        self.fc =  LinearLayer(128*W*W,512,droprate=0.0)
        # self.classifier = nn.Linear(512,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        # x = self.classifier(x)
        return x

class fc_rec(nn.Module):
    def __init__(self,  num_classes = 11):
        super(fc_rec, self).__init__()  
        self.fc =  LinearLayer(512,256,droprate=0.0)
        self.fc1 =  LinearLayer(256,128,droprate=0.0)
        self.classifier = nn.Linear(128,num_classes)

    def forward(self, input):
        x = self.fc(input)
        x = self.fc1(x)
        x = self.classifier(x)
        return x

class wirelss_ch(nn.Module):
    def __init__(self):
        super(wirelss_ch, self).__init__()  
        self.noise = None
        self.weights = None
        self.h = None
        self.n = None

    def forward(self, input):
        # # if self.h == None:
        self.h = torch.randn_like(input[0:1], dtype=torch.cfloat).to('cuda')
        # h_r = torch.randn_like(input).to('cuda')/2
        # h_i = 1j*torch.randn_like(input).to('cuda')/2
        # h = (h_r+h_i)/10
        # if self.n == None:
        self.n = torch.randn_like(input[0:1]).to('cuda')*1e-3
        phase =  2*torch.pi * input
        carrier = torch.exp(1j * phase)
        snr_db = 5
        snr_watt = 10**(snr_db/10)
        power_gain  = (snr_watt*1e-6/2)**0.5
        x = power_gain*carrier*(phase>0).to(torch.float32) + self.n*(torch.conj(self.h))/(torch.abs(self.h)**2)
        # x = carrier*self.h + self.n
        x = torch.cat((x.real, x.imag),dim=-1)
        # x = torch.cat((input, input),dim=-1)
        return x
        # if self.weights is None:
        #     self.weights = torch.randn(input.size()[-1],input.size()[-1], dtype=torch.cfloat).to('cuda')
        # h = self.weights 
        # new_dim = [int(input.shape[1]*self.T),]
        # new_dim.extend(input.shape[2:])
        # input = input.reshape(new_dim)
        # if self.noise is None:
        #     self.noise = torch.randn_like(input[0]).unsqueeze(0)
        # x = torch.matmul(input.to(torch.cfloat), h)+self.noise
        # x = torch.cat((x.real, x.imag),dim=-1)
        # batch_size = int(x.shape[0]/self.T)
        # new_dim = [self.T, batch_size]
        # new_dim.extend(x.shape[1:])
        # return x.reshape(new_dim)   

from snncutoff.gradients import GZIF


class wirelss_ch_odfm(nn.Module):
    def __init__(self,E,
                 neuron_params: dict = {'vthr': 1.0}):
        super(wirelss_ch_odfm, self).__init__()  
        self.noise = None
        self.weights = None
        self.h = None
        self.n = None
        self.group_size = 8
        self.num_path = 32
        self.E = E  # average energy consumption of per data subcarrier
        self.num_bit = neuron_params.num_bit
        # num_bit = num_bit if num_bit < 6 else 6
        # self.S = nn.Parameter(torch.tensor(3.0).float(), requires_grad=False)
        self.first_epoch = True
        # self.S = nn.Parameter(torch.tensor(3.0).float(), requires_grad=True)
        # self.min = nn.Parameter(torch.tensor(0.5).float(), requires_grad=True)
        self.surrogate = GZIF.apply
        self.momentum  = 0.9
        self.modulation = neuron_params.modulation
        self.L = 2
        self.pilot_interval = 8
        self.pilotValue = math.sqrt(10 ** (E / 10))
        self.B = neuron_params.B  # number of bits per subcarrier. Set to be an even number to be supported by QAM
        self.code_rate = 0.5
        self.num_ofdma = neuron_params.num_ofdma
        self.block_power_constrs = False
        self.power_constrs = 'block'
        print('power_constrs:',self.power_constrs)

    def forward(self, input):
        if self.modulation=='analog':
            return self._forward_analog(input)
        elif self.modulation=='noiseless':
            return self._forward_noiseless(input)
        else:
            return self._forward_digital(input)
    def _forward_noiseless(self, input):
        return input
    def _forward_digital(self, input):
        #input shape (time_step, batch_size, num_neurons)

        # num_spike_per_time * (m+log2{N})/r is the total number of bits to be transmitted for all the generated graded spikes at each time step
        # B*N is the total number of bits that can be carried by an OFDM symbol. Therefore, num_spike_per_time * (m+log2{N})/(rBN) is the number of OFDM symbols needed to transmit all spikes

        # the function below implements AER representation for each graded spike. The shape is (B,T,num_max_spiking_neurons_over_all_time_and_batch, ceil(log2(K))+m). At some time step, we pad the rows with all -1 to make it align with the dimension of num_max_spiking_neurons_over_all_time
        AER_symbol=self._convert_nonzero_to_log2D_m_bits(input.transpose(0,1)*(2**self.num_bit)-1, self.num_bit)  # (batch_size, time_step, num_max_spiking_neurons_over_all_time, ceil(log2(num_neurons))+m)
        num_max_spike = AER_symbol.size(-2)
        # print(num_max_spike)
        AER_symbol = tf.convert_to_tensor(AER_symbol.cpu().numpy() , dtype=tf.float32)

        num_add = tf.math.ceil(tf.math.log(tf.cast(input.shape[2], tf.float32)) / tf.math.log(2.0))  # log2(num_neurons)
        # self.code_rate = self.num_spike_supported * (num_add+self.num_bit) / (self.B * input.shape[2]) 
        encoder = sn.fec.ldpc.LDPC5GEncoder(self.B * input.shape[2]*self.code_rate, self.B * input.shape[2])
        decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

        # The number of subcarriers equals to the number of output neurons. Given the number of encoded bits, calculate the number of bits to be allocated to each subcarrier.
        constellation = sn.mapping.Constellation("qam", self.B, normalize=True)
        mapper = sn.mapping.Mapper(constellation=constellation)
        demapper = sn.mapping.Demapper("app", constellation=constellation)

        input_decoding_snn = -1 * torch.ones_like(input) # (batch_size, time_step, num_neurons)
        for t in range(input.size(0)):
            stream_m_batch = AER_symbol[:,t,:,:]  # (batch_size, num_max_spiking_neurons_over_all_time_and_batch, ceil(log2(num_neurons))+m)
            # print(stream_m_batch.shape[1])

            all_minus_1 = tf.reduce_all(stream_m_batch == -1, axis=-1)
            not_all_minus_1 = tf.logical_not(all_minus_1)
            row_counts = tf.reduce_sum(tf.cast(not_all_minus_1, tf.int32), axis=1)  # number of graded spikes per time for each data in the batch
            # The maximal value in 'row_counts' is <= than 'num_max_spiking_neurons_over_all_time_and_batch', 
            # since 'max{row_counts}' is the maximal number of graded spikes at this time, while 'num_max_spiking_neurons_over_all_time_and_batch' is across all time and batch
            
            # The number of ofdm symbols needed for each data in the batch for all the produced spikes at this time step
            num_ofdm = tf.math.ceil(tf.cast(row_counts, tf.float32) *(num_add+self.num_bit) /(self.code_rate*self.B*input.shape[2]))
            # print(num_ofdm)

            stream_m_batch = stream_m_batch[:, :tf.reduce_max(row_counts), :]
            # Then replace the elements of those rows with all -1 with all 0.
            all_minus_one_mask = tf.reduce_all(tf.equal(stream_m_batch, -1), axis=-1, keepdims=True)  # (batch_size, max(row_counts), 1)
            stream_m_batch = tf.where(all_minus_one_mask, tf.zeros_like(stream_m_batch), stream_m_batch)  # (batch_size, max(row_counts), ceil(log2(num_neurons))+m)
            stream_m = tf.reshape(stream_m_batch, (input.shape[1], -1))  # (batch_size, max(row_counts) * (ceil(log2(num_neurons))+m) )

            # pad each data stream in 'stream_m' to have max{num_ofdm}*(code_rate*B*N) bits
            pad_amount = tf.reduce_max(num_ofdm)*self.code_rate*self.B*input.shape[2] - stream_m.shape[-1]
            paddings = [[0, 0], [0, pad_amount]]
            stream_m = tf.pad(stream_m, paddings) if pad_amount > 0 else stream_m   # (batch_size, max_num_ofdm * code_rate*B*N )
            # print(stream_m.shape[1]/tf.reduce_max(num_ofdm))
            # prepare it to several ofdm 
            stream_m = tf.reshape(stream_m, [tf.shape(stream_m)[0], tf.cast(tf.reduce_max(num_ofdm), tf.int32), -1]) # (batch_size, max_num_ofdm, code_rate*B*N)

            encoded_bits = encoder(stream_m).numpy() # (batch_size, max_num_ofdm, B*N) 
            # print(encoded_bits.shape)

            PAM_symbol = mapper(encoded_bits)  # (batch_size, max_num_ofdm, N) 
            # print(PAM_symbol.shape)

            # calculate the power scaling of shape (batch_size, max_num_ofdm)
            E_watt = 10 ** (self.E / 10)  # transform db to Watt
            # gamma = PAM_symbol.shape[2] * E_watt/tf.reduce_sum(tf.square(tf.abs(PAM_symbol)), axis=2)  # (batch_size, max_num_ofdm)
            # gamma = tf.cast(gamma, tf.complex64)

            if self.power_constrs == 'block':
                gamma = PAM_symbol.shape[2] * E_watt/tf.reduce_sum(tf.square(tf.abs(PAM_symbol)), axis=2)  # (batch_size, max_num_ofdm)
                # tf.print(gamma[:,0])
                gamma = tf.cast(gamma, tf.complex64)
            elif self.power_constrs == 'peak':
                gamma = E_watt / tf.reduce_max((tf.abs(PAM_symbol) ** 2),axis=-1)
                gamma = tf.cast(gamma, tf.complex64)


            # print(tf.reduce_sum(tf.square(tf.abs(PAM_symbol))))
            PAM_symbol = PAM_symbol * tf.sqrt(gamma)[:, :, tf.newaxis] # (batch_size, max_num_ofdm, N) 
            # print(tf.reduce_sum(tf.square(tf.abs(PAM_symbol)))/PAM_symbol.shape[1])  # should be equal to E
                            
            # construct the data and pilot subcarriers, both are 1D tensor containing the indices of data and pilot subcarriers respectively
            num_data_carriers = tf.shape(PAM_symbol)[2]
            batch_size = tf.shape(PAM_symbol)[0]
            num_pilot = tf.cast(tf.math.ceil(tf.cast(num_data_carriers, tf.float32) / (self.pilot_interval - 1)), tf.int32)
            num_allcarriers = num_data_carriers + num_pilot
            allCarriers = tf.range(num_allcarriers, dtype=tf.int32)
            pilotCarriers = tf.range(0, num_allcarriers, self.pilot_interval, dtype=tf.int32)
            if pilotCarriers[-1] != allCarriers[-1]:
                # Append num_allcarriers to allCarriers
                allCarriers = tf.concat([allCarriers, tf.stack([num_allcarriers])], axis=0)
                num_allcarriers += 1
                # Append the last element to pilotCarriers
                pilotCarriers = tf.concat([pilotCarriers, tf.stack([num_allcarriers-1])], axis=0)
                num_pilot += 1
            # Create a boolean mask to check if elements of allCarriers are in pilotCarriers
            mask = tf.reduce_any(tf.equal(tf.expand_dims(allCarriers, axis=1), pilotCarriers), axis=1)
            # Use the mask to filter out the elements in allCarriers that are in pilotCarriers
            dataCarriers = tf.boolean_mask(allCarriers, ~mask)

            # construct the OFDM symbol based on the subcarriers
            max_num_ofdm = tf.cast(tf.reduce_max(num_ofdm), tf.int32)
            if self.num_ofdma == 100:
                self.num_ofdma = max_num_ofdm.numpy().item()
            OFDM_data = tf.zeros((batch_size, max_num_ofdm, num_allcarriers), dtype=tf.complex64)   # (batch_size, max_num_ofdm, num_all_subcarrier)
            # Generate a grid of batch and pilot indices
            batch_grid, num_ofdm_grid, pilot_grid = tf.meshgrid(tf.range(batch_size), tf.range(max_num_ofdm), pilotCarriers, indexing='ij')
            # Stack and reshape indices to match the required shape for scatter_nd_update
            indices = tf.stack([tf.reshape(batch_grid, [-1]), tf.reshape(num_ofdm_grid, [-1]), tf.reshape(pilot_grid, [-1])], axis=1)
            # Prepare the values to assign, matching the shape of indices
            values = tf.fill([tf.shape(indices)[0]], tf.cast(self.pilotValue, tf.complex64))
            # Update OFDM_data at the specified indices with the pilot values
            OFDM_data = tf.tensor_scatter_nd_update(OFDM_data, indices, values)

            batch_grid, num_ofdm_grid, data_grid = tf.meshgrid(tf.range(batch_size), tf.range(max_num_ofdm), dataCarriers, indexing='ij')
            indices = tf.stack([tf.reshape(batch_grid, [-1]), tf.reshape(num_ofdm_grid, [-1]), tf.reshape(data_grid, [-1])], axis=1)
            values = tf.reshape(PAM_symbol, [-1])
            values = tf.cast(values, tf.complex64)
            OFDM_data = tf.tensor_scatter_nd_update(OFDM_data, indices, values)
            # print(OFDM_data)
            
            channelResponse = tf.complex(tf.random.normal((batch_size, max_num_ofdm, self.num_path), 0.0, math.sqrt(1/2)), tf.random.normal((batch_size, max_num_ofdm, self.num_path), 0.0, math.sqrt(1/2)))
            # Compute the FFT along the last axis (by default)
            H_exact = tf.signal.fft(tf.pad(channelResponse, [[0, 0], [0, 0], [0, num_allcarriers - self.num_path]]))  # Shape: (batch_size, max_num_ofdm, num_allcarriers)

            noise_power = 1e-3
            channelNoise = tf.complex(tf.random.normal((batch_size, max_num_ofdm, num_allcarriers), 0.0, noise_power), tf.random.normal((batch_size, max_num_ofdm, num_allcarriers), 0.0, noise_power))
            OFDM_demod = OFDM_data * H_exact + channelNoise
            pilots = tf.gather(OFDM_demod, pilotCarriers, axis=2)  # Shape: (batch_size, max_num_ofdm, num_pilots)
            Hest_at_pilots = pilots / self.pilotValue  # Shape: (batch_size, max_num_ofdm, num_pilots)

            Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)  # interpolation to obtain all subcarriers' amplitude
            Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)  # interpolation to obtain all subcarriers' phase
            Hest = Hest_abs * tf.math.exp(1j*Hest_phase)  # (batch_size, max_num_ofdm, num_allcarriers)
            Hest = tf.cast(Hest,dtype=tf.complex64)
            equalized_Hest = OFDM_demod / Hest  # equalization
            equalized_data = tf.gather(equalized_Hest, dataCarriers, axis=2)/tf.expand_dims(tf.sqrt(gamma), axis=-1) # equalized data symbols (batch_size, max_num_ofdm, num_data_carriers)
            # print(equalized_data.shape)
            

            ebno_db = 10 * np.log10(E_watt/self.B/noise_power)
            no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.B, coderate=self.code_rate)
            llr = demapper([equalized_data, no])  # (batch_size, max_num_ofdm, num_data_carriers*B)
            # print(llr.shape, encoded_bits.shape)
            
            bits_hat = decoder(llr)  # (batch_size, max_num_ofdm, num_data_carriers*B*code_rate)
            bits_hat = tf.reshape(bits_hat, [tf.shape(bits_hat)[0], -1])  # (batch_size, max_num_ofdm * num_data_carriers*B*code_rate)
            original_width = tf.reduce_max(num_ofdm) * self.code_rate * self.B * input.shape[2] - pad_amount
            bits_hat = bits_hat[:, :tf.cast(original_width, tf.int32)]  # (batch_size, max(row_counts) * (ceil(log2(num_neurons))+m) )
            bits_hat = tf.reshape(bits_hat, [tf.shape(bits_hat)[0], tf.reduce_max(row_counts), -1])  # (batch_size, max(row_counts), ceil(log2(num_neurons))+m )
            # max(row_counts) is the maximal number of spike at this batch of data at this time
            # The number of bits one ofdm symbol is able to support for orignial spikes: code_rate*B*N
            bits_hat_flat = tf.reshape(bits_hat, (batch_size, -1))
            valid_shape = tf.cast(self.num_ofdma * (self.code_rate * self.B * input.shape[2]), tf.int32)  # number of bits self.num_ofdma ofdm symbols are able to support for orignial spikes
            remaining_shape = tf.cast(bits_hat.shape[1] * bits_hat.shape[2], tf.int32) - valid_shape
            if remaining_shape>0:
                mask = tf.concat([tf.ones((batch_size, valid_shape), dtype=tf.float32),tf.zeros((batch_size, remaining_shape), dtype=tf.float32)], axis=1)
                bits_hat_flat = bits_hat_flat*mask
            
            bits_hat_tf = tf.reshape(bits_hat_flat, (batch_size, bits_hat.shape[1], -1))  # (batch_size, max(row_counts), ceil(log2(num_neurons))+m )
            
            # Step 1: Extract address bits and information bits
            address_bits = bits_hat_tf[:, :, :tf.cast(num_add, tf.int32)]   # Shape: [batch_size, max(row_counts), num_add]
            info_bits = bits_hat_tf[:, :, tf.cast(num_add, tf.int32):]      # Shape: [batch_size, max(row_counts), num_bit]
            # Convert address bits and info bits to decimal values
            address_values = self._binary_to_decimal(address_bits, num_add)  # Shape: [batch_size, max(row_counts)]
            info_values = self._binary_to_decimal(info_bits, self.num_bit)        # Shape: [batch_size, max(row_counts)]
            address_torch = torch.from_numpy(address_values.numpy()).long().to(input.device)
            info_torch = torch.from_numpy(info_values.numpy()).to(dtype=input_decoding_snn.dtype).to(input.device)
            input_decoding_snn[t].scatter_(dim=1, index=address_torch, src=info_torch)

            # For checking purpose
            # print ("Bit recovery rate: ", tf.reduce_mean(tf.cast(tf.equal(stream_m, bits_hat), tf.float32)))
        # For checking purpose
        # print((input == ((input_decoding_snn+1)/(2**self.num_bit))).sum().item()/input.numel())

        return (input_decoding_snn+1)/(2**self.num_bit) 
    
    def _binary_to_decimal(self, bits, num_bits):
        """
        Converts binary bits to decimal values.
        Args:
            bits: Tensor of shape [batch_size, num_spike_supported, num_bits]
            num_bits: Number of bits in the binary representation
        Returns:
            decimals: Tensor of shape [batch_size, num_spike_supported]
        """
        # Step 0: Ensure bits are of integer type
        bits = tf.cast(bits, tf.int32)

        # Step 1: Create exponents for the binary digits (most significant bit to least significant bit)
        exponents = tf.range(num_bits - 1, -1, -1, dtype=tf.int32)  # Shape: [num_bits]

        # Step 2: Compute powers of two
        powers_of_two = tf.pow(2, exponents)  # Shape: [num_bits], dtype: tf.int32

        # Step 3: Reshape for broadcasting
        powers_of_two = tf.reshape(powers_of_two, [1, 1, num_bits])  # Shape: [1, 1, num_bits]

        # Step 4: Multiply bits by powers of two and sum over the bits
        decimals = tf.reduce_sum(bits * powers_of_two, axis=-1)  # Shape: [batch_size, num_spike_supported]

        return decimals


    # This function that takes the output of the encoding SNN of shape (batch_size, time_step, num_neurons), in which each elements take a value from {0,1,2,....,2^m}, and then produces a matrix of shape (batch_size,time_step,num_max_spiking_neurons_over_all_time, ceil(log2(K))+m). To align with the shape, those idle neurons will be enecoded into all -1 
    def _convert_nonzero_to_log2D_m_bits(self, tensor, m):
        tensor = tensor.to(torch.int)
        batch_size, T, D = tensor.shape
        
        # Determine log2D
        log2D = torch.ceil(torch.log2(torch.tensor(D, device=tensor.device))).to(torch.int32)
        
        # Step 1: Create a mask for valid elements (values from 0 to 2^m, ignore -1)
        valid_mask = tensor.ge(0)  # Mask for elements to encode, where True indicates valid elements

        # Step 2: Extract m bits for valid elements (0, 1, ..., 2^m)
        mask_m = 2 ** torch.arange(m - 1, -1, -1, device=tensor.device).unsqueeze(0)
        m_bits = (tensor.unsqueeze(-1) & mask_m).ne(0).int()  # Shape: (batch_size, T, D, m)

        # Step 3: Get the binary indices for the D dimension (log2(D) bits)
        neuron_indices = torch.arange(D, device=tensor.device)  # Shape: (D,)
        mask_log2D = 2 ** torch.arange(log2D - 1, -1, -1, device=tensor.device).unsqueeze(0)
        neuron_bits = (neuron_indices.unsqueeze(-1) & mask_log2D).ne(0).int()  # Shape: (D, log2(D))

        # Step 4: Expand the neuron bits to match the tensor shape
        neuron_bits_expanded = neuron_bits.unsqueeze(0).unsqueeze(0).expand(batch_size, T, D, log2D)

        # Step 5: Concatenate log2(D) bits first and then the m bits for valid elements
        log2D_m_bits = torch.cat((neuron_bits_expanded, m_bits), dim=-1)  # Shape: (batch_size, T, D, log2(D) + m)

        # Step 6: Gather only the bits for valid elements using the valid_mask
        valid_indices = valid_mask.nonzero(as_tuple=True)  # Shape: (num_valid_elements, 3)

        # Only gather bits where the elements are valid (i.e., >= 0)
        valid_bits = log2D_m_bits[valid_indices]  # Shape: (num_valid_elements, log2(D) + m)

        # Step 7: Prepare the output for each time step, only for valid elements
        valid_count_per_bt = valid_mask.sum(dim=-1)  # Number of valid elements per (batch, T)
        max_valid_count = valid_count_per_bt.max()  # Maximum number of valid elements per (batch, T)

        # Initialize the result tensor to store valid elements only
        encoded_result = -1 * torch.ones((batch_size, T, max_valid_count, log2D + m), dtype=torch.int, device=tensor.device)

        # Use advanced indexing to place the valid bits into the output tensor
        batch_idx, time_idx, _ = valid_indices  # Unpack indices ignoring the neuron dimension

        # Position index to place valid elements in the result tensor
        position_idx = torch.cumsum(valid_mask, dim=-1) - 1  # Create a cumulative count of valid entries
        position_idx = position_idx[valid_mask]  # Extract only valid positions
        
        # Place the valid bits into the result tensor
        encoded_result[batch_idx, time_idx, position_idx] = valid_bits

        return encoded_result


    def _forward_analog(self, input):

        pam_points = self._generate_mbit_pam(self.num_bit,device=input.device)
        #input shape (time_step, batch_size, num_neurons)
        input = input.transpose(0, 1)*(2**self.num_bit)  # (batch_size, time_step, num_neurons)

        # The number of data subcarriers is the number of output neurons
        batch_size = input.shape[0] 
        num_data_carriers = input.shape[2]
        num_pilot = int(torch.ceil(torch.tensor(num_data_carriers / (self.pilot_interval - 1))))
        num_allcarriers = num_data_carriers + num_pilot
        allCarriers = torch.arange(num_allcarriers,device=input.device)
        pilotCarriers = torch.arange(0, num_allcarriers, self.pilot_interval,device=input.device)
        if pilotCarriers[-1] != allCarriers[-1]:
            allCarriers = torch.cat([allCarriers, torch.tensor([num_allcarriers]).to(input.device) ])
            num_allcarriers += 1
            pilotCarriers = torch.cat([pilotCarriers, torch.tensor([allCarriers[-1]]).to(input.device) ])
            num_pilot += 1
        dataCarriers = allCarriers[~torch.isin(allCarriers, pilotCarriers)]

        # prepare for interpolation
        lower_indices = torch.searchsorted(pilotCarriers, dataCarriers, right=True) - 1
        upper_indices = lower_indices + 1
        lower_indices = torch.clamp(lower_indices, 0, num_pilot - 1).to(input.device) 
        upper_indices = torch.clamp(upper_indices, 0, num_pilot - 1).to(input.device) 
        weights = (dataCarriers.float() - pilotCarriers[lower_indices].float()) / (pilotCarriers[upper_indices].float() - pilotCarriers[lower_indices].float())

        constellation_points = torch.arange(2 ** self.num_bit + 1, dtype=torch.float32).view(1, -1).to(input.device) 
        temperature = 0.0001  # You can adjust this value
        input_decoding_snn = torch.zeros_like(input).transpose(0,1) # (time_step, batch_size, num_neurons)

        for t in range(input.size(1)):
            OFDM_data = torch.zeros((batch_size, self.num_ofdma, num_allcarriers), dtype=torch.complex64,device=input.device)
 
            OFDM_data[:, :, pilotCarriers] = self.pilotValue * torch.ones_like(OFDM_data[:, :, pilotCarriers],device=input.device)  # assign a known pilot symbol to all pilot subcarriers in an OFDM symbol

            # calculate the power scaling 
            E_watt = 10 ** (self.E / 10)  # transform db to Watt
            pam_values = pam_points[input[:, t, :].long()] + 0.j  # (batch_size, num_neurons)

            # gamma = num_data_carriers * E_watt/torch.sum(pam_values ** 2, dim=1,keepdim=True)  # (batch_size,)
            # print(E_watt)
            # print(gamma*torch.sum(input[:,t,:] ** 2, dim=1)/num_data_carriers)
            if self.power_constrs == 'block':
                gamma = num_data_carriers * E_watt/torch.sum(pam_values.abs() ** 2, dim=1,keepdim=True)
            elif self.power_constrs == 'peak':
                gamma = E_watt / torch.abs(pam_values).pow(2).max(dim=-1,keepdim=True)[0]

            OFDM_data[:, :, dataCarriers] = (pam_values*torch.sqrt(gamma)).to(torch.complex64).unsqueeze(1).expand(-1, self.num_ofdma, -1).to(input.device)# assign all the data symbols to their corresponding data subcarriers in an OFDM symbol


            channelResponse = torch.normal(mean=0, std=math.sqrt(1/2), size=(batch_size, self.num_ofdma, self.num_path),device=input.device) +1j * torch.normal(mean=0, std=math.sqrt(1/2), size=(batch_size, self.num_ofdma, self.num_path),device=input.device) # the impulse response of the wireless channel
            H_exact = torch.fft.fft(channelResponse, n=num_allcarriers, dim=2)  # (batch_size, num_ofdm, num_allcarriers)

            noise_power = 1e-3
            OFDM_demod = OFDM_data * H_exact + torch.normal(mean=0, std=noise_power, size=(batch_size, self.num_ofdma, num_allcarriers),device=input.device)

            pilots = OFDM_demod[:, :, pilotCarriers] # (batch_size, num_ofdm, num_pilots)
            Hest_at_pilots = pilots / self.pilotValue  # (batch_size, num_ofdm, num_pilots)

            Hest_abs = torch.lerp(
                torch.abs(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), lower_indices]),
                torch.abs(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), upper_indices]),
                weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_data_carriers)
            )
            # Interpolation for the phases of the channel estimates
            Hest_phase = torch.lerp(
                torch.angle(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), lower_indices]),
                torch.angle(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), upper_indices]),
                weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_data_carriers)
            )
            Hest = Hest_abs * torch.exp(1j * Hest_phase)  # (batch_size, num_ofdm, num_data_carriers)

            equalized_data = OFDM_demod[:, :, dataCarriers] / Hest /torch.sqrt(gamma).unsqueeze(2) # equalized data symbols (batch_size, num_ofdm, num_data_carriers)
            # MRC for the equalized data
            combined_OFDM = equalized_data.sum(dim=1)  # Shape: (batch_size, num_carriers)
            mrc_output = combined_OFDM / self.num_ofdma  # Shape: (batch_size, num_carriers)

            distances = torch.abs(mrc_output.unsqueeze(-1) - pam_points)  # (batch_size, num_data_carriers, 2^num_bits+1)
            if self.training:
                soft_probs = torch.softmax(-distances / temperature, dim=-1)  # Shape: (batch_size, num_data_carriers, 2^num_bits+1)
                symbol_indices = torch.arange(2 ** self.num_bit, device=distances.device).expand(distances.size(0), distances.size(1), -1)  # Shape: (batch_size, num_data_carriers, 2^num_bits+1)
                closest_symbols = torch.sum(soft_probs * symbol_indices, dim=-1)  # Shape: (batch_size, num_data_carriers)
            else:
                closest_symbols = torch.argmin(distances, dim=-1)  # (batch_size, num_data_carriers)

            # print ("Recovery rate: ", torch.sum(torch.eq(closest_symbols, input[:,t,:]))/(batch_size*num_data_carriers))
            input_decoding_snn[t] = closest_symbols
      
        return input_decoding_snn/(2**self.num_bit)

    def _generate_mbit_pam(self, m, device):
        if m == 0:
            # For m = 0, we only need two points: 0 and 1
            constellation_points = torch.tensor([0, 1], dtype=torch.float32, device=device)
        # elif m == 1:
        #     # For m = 1, we need three points: 0, -1, and 1
        #     constellation_points = torch.tensor([0, -1, 1], dtype=torch.float32, device=device)
        else:
            # For m >= 2, we need 2^m points with zero at the first index
                # Determine the number of levels
            L = 2 ** m
            
            # Generate unnormalized PAM levels (symmetrical around 0)
            levels = torch.arange(-L + 1, L, 2, device=device, dtype=torch.float32)
            
            # Calculate the mean square value of unnormalized levels
            mean_square = torch.mean(levels ** 2)
            
            # Normalize the levels so that the mean square equals 1
            constellation_points = levels / torch.sqrt(mean_square)
            constellation_points = torch.cat([torch.tensor([0.0], device=device), constellation_points])

        return constellation_points

    def __forward_digital(self, input):
        input_size = input.size() 
        indices = torch.arange(input_size[-1]).unsqueeze(0).expand(input_size[0],input_size[1], -1).to(input.device)*2**self.L
        x_with_indices = torch.stack((input*2**self.L, indices), dim=-1)
        zero_mask = (input == 0).float().unsqueeze(-1)
        x_with_indices = x_with_indices*(1-zero_mask)
        # num_non_zero = (1-zero_mask).sum(-2)
        sorted_x_with_indices, _ = torch.sort(x_with_indices, dim=2, descending=True)
        sorted_x_with_indices = sorted_x_with_indices[...,0]+sorted_x_with_indices[...,1]
        # print(sorted_x_with_indices.max(dim=2)[0].int().log2().floor()+1)
        carrier_tilde_t = []
        for t in range(input_size[0]):
            carrier_tilde_b = []
            for i in range(input_size[1]):
                data = sorted_x_with_indices[t:t+1,i:i+1]
                bit_width  = data.max().int().log2().floor()+1
                powers_of_two = 2 ** torch.arange(int(bit_width.item()) - 1, -1, -1).long().to(input.device)
                binary_tensor = ((data.int().unsqueeze(-1) & powers_of_two) > 0).long()
                B = (data+1).int().log2().sum(dim=-1)//input.size()[-1]
                binary_tensor = binary_tensor.flatten(start_dim=-2)
                # binary_tensor = binary_tensor[...,:int(bit_width.item()*num_non_zero[t,i].item())]
                binary_tensor = binary_tensor[...,:int(B.item()*input_size[-1])]
                binary_tensor = binary_tensor.reshape([1,1,input_size[-1],int(B.item())])
                powers_of_two = 2 ** torch.arange(int(B.item()) - 1, -1, -1).to(binary_tensor.device).float()
                analog_data = (binary_tensor * powers_of_two).sum(dim=-1)/2**B
                gamma = self.E/(analog_data.pow(2).mean(-1,keepdim=True).detach()+self.E*1e-3)
                carrier = gamma.sqrt()*analog_data

                num_groups = carrier.size(-1)//self.group_size
                new_dim = list(carrier.size()[:-1])
                new_dim.extend([num_groups,self.group_size])
                carrier = carrier.reshape(new_dim)
                pilot = torch.ones_like(carrier[...,0:1])*torch.tensor(self.E).sqrt()
                carrier = torch.cat((pilot,carrier), dim=-1)
                new_dim = list(carrier.size()[:-2])
                new_dim.extend([int(carrier.size()[-2]*carrier.size()[-1])])
                carrier = carrier.reshape(new_dim)

                #Simulate the channel impulse response 
                h = torch.randn((carrier.size(0),carrier.size(1), self.num_path), dtype=torch.cfloat, device=input.device)  # 5-path channel
                padding = (0, carrier.size()[-1] -  self.num_path)
                # Apply the padding
                h = F.pad(h, padding)
                h = torch.fft.fft(h, dim=-1)
                noise = torch.randn_like(carrier, dtype=torch.cfloat) * 1e-3
                #forwarded to wireless channel
                carrier_tilde = carrier + noise*torch.conj(h)/(torch.abs(h)**2)

         

                #remove pilot
                num_groups = carrier_tilde.size(-1)//(self.group_size+1)
                new_dim = list(carrier_tilde.size()[:-1])
                new_dim.extend([num_groups,self.group_size+1])
                carrier_tilde = carrier_tilde.reshape(new_dim)
                carrier_tilde = carrier_tilde[...,1:]
                new_dim = list(carrier_tilde.size()[:-2])
                new_dim.extend([int(carrier_tilde.size()[-2]*carrier_tilde.size()[-1])])
                carrier_tilde = carrier_tilde.reshape(new_dim).real
                #quantization
                L = 2**self.num_bit-1
                # carrier_tilde = self.surrogate((B*carrier_tilde/self.S.exp()+self.min),L,self.S)/B
                # carrier_tilde = torch.clamp(carrier_tilde,0.0,1.0)
                carrier_tilde_b.append(carrier_tilde[0,0])
            carrier_tilde_b = torch.stack(carrier_tilde_b,dim=0)
            carrier_tilde_t.append(carrier_tilde_b)
        carrier_tilde = torch.stack(carrier_tilde_t,dim=0)
        print(carrier_tilde.shape)
        # bit_width  = sorted_x_with_indices.max().int().log2().floor()+1
        # powers_of_two = 2 ** torch.arange(int(bit_width.item()) - 1, -1, -1).long().to(input.device)
        # binary_tensor = ((sorted_x_with_indices.int().unsqueeze(-1) & powers_of_two) > 0).long()
        # B = (sorted_x_with_indices+1).int().log2().sum(dim=-1)//input.size()[-1]
        # First, ensure the bit width is divisible by 3 by padding the binary tensor if needed


        # Reshape to split into 3-bit groups
        # print(binary_flattened.shape)

        # print(sorted_x_with_indices[0,0,...,0])
        # print(sorted_x_with_indices[0,0,...,1])
        # print(sorted_x_with_indices.max().int(),sorted_x_with_indices.max(),B)

        # data_hat = input[non_zero_indices]  # Non-zero positive values
        # addresses = non_zero_indices[1]  # Corresponding indices within the data
        # batch_indices = non_zero_indices[0]  # Batch indices
        # max_non_zeros = non_zero_mask.sum(dim=1).max().item()
        # hat_x = torch.zeros((input.size(1), max_non_zeros, 2))
        # print(addresses.shape)
        
        return carrier_tilde

    def __forward_analog(self, input):
        # Power Normalization
        carrier = input
        gamma = self.E/(input.pow(2).mean(-1,keepdim=True).detach()+self.E*1e-3)
        carrier = gamma.sqrt()*carrier

        #Add Pilot
        num_groups = carrier.size(-1)//self.group_size
        new_dim = list(carrier.size()[:-1])
        new_dim.extend([num_groups,self.group_size])
        carrier = carrier.reshape(new_dim)
        pilot = torch.ones_like(carrier[...,0:1])*torch.tensor(self.E).sqrt()
        carrier = torch.cat((pilot,carrier), dim=-1)
        new_dim = list(carrier.size()[:-2])
        new_dim.extend([int(carrier.size()[-2]*carrier.size()[-1])])
        carrier = carrier.reshape(new_dim)

        #Simulate the channel impulse response 
        h = torch.randn((carrier.size(0),carrier.size(1), self.num_path), dtype=torch.cfloat, device=input.device)  # 5-path channel
        padding = (0, carrier.size()[-1] -  self.num_path)
        # Apply the padding
        h = F.pad(h, padding)
        h = torch.fft.fft(h, dim=-1)
        noise = torch.randn_like(carrier, dtype=torch.cfloat) * 1e-3

        #forwarded to wireless channel
        carrier_tilde = carrier + noise*torch.conj(h)/(torch.abs(h)**2)

        #remove pilot
        num_groups = carrier_tilde.size(-1)//(self.group_size+1)
        new_dim = list(carrier_tilde.size()[:-1])
        new_dim.extend([num_groups,self.group_size+1])
        carrier_tilde = carrier_tilde.reshape(new_dim)
        carrier_tilde = carrier_tilde[...,1:]
        new_dim = list(carrier_tilde.size()[:-2])
        new_dim.extend([int(carrier_tilde.size()[-2]*carrier_tilde.size()[-1])])
        carrier_tilde = carrier_tilde.reshape(new_dim).real
        
        #quantization
        L = 2**self.num_bit-1
        carrier_tilde = self.surrogate((L*carrier_tilde/self.S.exp()+self.min),L,self.S)/L
        carrier_tilde = torch.clamp(carrier_tilde,0.0,1.0)
  
        return carrier_tilde

        # gamma = 1/(carrier_tilde.pow(2).mean(-1,keepdim=True).detach()+1e-3)
        # carrier_tilde = gamma.sqrt()*carrier_tilde
        # carrier_tilde = carrier_tilde-carrier_tilde.min()
        # if self.training:
        #     if self.first_epoch:
        #         a =  carrier_tilde.detach().sum()/(carrier_tilde>0).float().sum()*2
        #         self.S.data = a
        #         self.first_epoch = False
        #     a =  carrier_tilde.detach().sum()/(carrier_tilde>0).float().sum()*2
        #     self.S.data = self.momentum * self.S.data + (1 - self.momentum) * a

        # print(self.max,self.min)
        # if self.training:
        #     if self.first_epoch:
        #         self.max.data = torch.log(carrier_tilde.max())
        #         self.first_epoch = False
            # self.max.data = self.momentum * self.max.data + (1 - self.momentum) * carrier_tilde.max()
        # carrier_tilde = (carrier_tilde)/(self.max.exp()+1e-5)
        # print(self.max.exp())
      #quantization 
        # print(carrier_tilde.shape)
# class VGG_NeuroComm(nn.Module):
#     def __init__(self,output_dim = 11):
#         super(VGG_Gesture, self).__init__()
#         self.encoder = VGG_Encoder
#         self.decoder = VGG_Decoder
#         self.wirelss_ch = wirelss_ch

#     def forward(self, input):
#         x = self.encoder(input)
#         x = self.wirelss_ch(x)
#         x = self.decoder(x)
#         return x
from snncutoff.constrs.utils.pre_constrs import PreConstrs
from snncutoff.constrs.utils.post_constrs import PostConstrs

class VGG_NeuroComm_eval(nn.Module):
    def __init__(self,num_classes = 11,E=1,
                 neuron_params: dict = {'vthr': 1.0},):
        super(VGG_NeuroComm_eval, self).__init__()
        # pool = nn.MaxPool2d(2)
        # self.features = nn.Sequential(
        #     Conv2dLayer(2,64,4,4,padding='valid'),
        #     Conv2dLayer(64,128,3,1,1),
        #     pool,
        #     Conv2dLayer(128,128,3,1,1),
        #     pool,
        #     Conv2dLayer(128,128,3,1,1),
        #     pool,
        #     Conv2dLayer(128,128,3,1,1),
        #     pool,
        #     Conv2dLayer(128,128,3,1,1),
        #     pool,
        #     nn.Flatten(1,-1)
        # )
        # W = int(128/4/2/2/2/2/2)
        # self.fc =  LinearLayer(128*W*W,512,droprate=0.0)
        self.features = VGG_Encoder()
        self.wireless_ch = wirelss_ch_odfm(E,neuron_params)
        self.classifier = fc_rec()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.wireless_ch(x)
        x = self.classifier(x)
        return x