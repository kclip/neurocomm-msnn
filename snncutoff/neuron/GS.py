import torch
import torch.nn as nn
from torch.autograd import Function
from snncutoff.gradients import GZIF,ZIF
from typing import Type


class GS(nn.Module):
    def __init__(self, T: int=4, 
                 vthr: float = 1.0, 
                 tau: float = 0.0, 
                 surogate: Type[Function] = GZIF,
                 mem_init: float = 0.5,
                 multistep: bool = True,
                 reset_mode: str ='hard',
                 num_bit: int=16,
                 **kwargs):
        
        super(GS, self).__init__()
        
        """
        Initialize the LIF neuron model.

        Args:
            vthr (float): The threshold voltage for spike generation.
            tau (float): The time constant of the membrane potential decay.
        """
        self.t = 0.0
        self.T = T
        self.vmem = 0.0
        self.vthr = vthr
        self.tau = tau
        self.gamma = 1.0
        self.reset_mode = reset_mode
        self.mem_init = mem_init
        self.multistep = multistep
        self.num_bit = num_bit
        if num_bit >= 0:
            self.surogate = surogate.apply
        else:
            self.surogate = ZIF.apply
        # num_bit = num_bit if num_bit < 6 else 6
        self.first_epoch = True
        if self.num_bit > 4:
            self.S = nn.Parameter(torch.tensor(-3).float(), requires_grad=True)
        elif self.num_bit >=2 and self.num_bit <= 4:
            self.S = nn.Parameter(torch.tensor(-1).float(), requires_grad=True)
        elif self.num_bit >=0 and self.num_bit < 2:
            self.S = nn.Parameter(torch.tensor(3.0).float(), requires_grad=True)
        else:
            self.S = nn.Parameter(torch.tensor(3).float(), requires_grad=False)
        self.momentum = 0.9

    def _mem_update_multistep(self, x):  
        spike_post = []
        mem_post = []
        self.reset()

        L = 2**self.num_bit if self.num_bit >= 0 else 1
        for t in range(self.T):
            vmem = self.vmem + x[t]
            if self.num_bit >= 0:
                vmem = vmem*self.S.sigmoid()
                spike = self.surogate(L*vmem,L)/L
            else:
                spike = vmem*self.surogate(L*vmem-self.vthr,L)/L
            vmem = self.vmem_reset(vmem,(spike>0).float())
            self.updateMem(vmem)
            spike_post.append(spike)
            mem_post.append(vmem)
        return torch.stack(spike_post,dim=0), torch.stack(mem_post,dim=0)

    def _mem_update_singlestep(self,x):
        L = 2**self.num_bit if self.num_bit >= 0 else 1
        if self.t == 0:
            self.mem_init = 0.5 if self.reset_mode == 'soft' else self.mem_init
            self.initMem(self.mem_init*self.vthr)
        spike_post = []
        vmem = self.vmem + x[0]
        # spike =  (vmem > self.vthr).float()
        if self.num_bit >= 0:
            vmem = vmem*self.S.sigmoid()
            spike = self.surogate(L*vmem,L)/L
        else:
            spike = vmem*self.surogate(L*vmem-self.vthr,L)/L
        vmem = self.vmem_reset(vmem,(spike>0).float())
        self.updateMem(vmem)
        spike_post.append(spike)
        return torch.stack(spike_post,dim=0), 0.0

    def forward(self,x):
        if self.multistep:
            return self._mem_update_multistep(x)
        else:
            return self._mem_update_singlestep(x)   


    def reset(self):
        """
        Reset the membrane potential and time step to initial values.
        """
        self.t = 0.0
        self.vmem = 0.0

    def initMem(self, x: float):
        """
        Initialize the membrane potential with a given value.

        Args:
            x (float): The initial membrane potential.
        """
        self.vmem = x

    def updateMem(self, x: float):
        """
        Update the membrane potential based on the input and time constant.

        Args:
            x (float): The input value to update the membrane potential.
        """
        self.vmem = x * self.tau
        self.t += 1

    def is_spike(self) -> bool:
        """
        Check if the membrane potential has reached the threshold.

        Returns:
            bool: True if the membrane potential has reached or exceeded the threshold, False otherwise.
        """
        return self.vmem >= self.vthr

    def vmem_reset(self, x, spike):
        if self.reset_mode == 'hard':
            return x * (1-spike)
        elif self.reset_mode == 'soft':  
            return x - self.vthr*spike

# Example usage:
# lif_neuron = LIF()
# lif_neuron.initMem(0.5)
# lif_neuron.updateMem(1.0)
# print(lif_neuron.is_spike())