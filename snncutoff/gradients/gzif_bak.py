import torch
from torch.autograd import Function

class GZIF(Function):
    # @staticmethod
    # def forward(ctx, input):
    #     return input.floor()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     if grad_output.min()<-1:
    #         print(grad_output.min(),grad_output.max())
    #     return grad_output


    @staticmethod
    def forward(ctx, input, L):
        # out = (input > 0).float()
        L = torch.tensor(L)
        out = input.floor().clamp(0.0,L)
        ctx.save_for_backward(input, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out) = ctx.saved_tensors
        
        # tmp = 0
        # tmp = 1
        grad_input = grad_output.clone()
        # if L == 1:
        #     tmp =  ((1 - (input-1/S).abs()).clamp(min=0))
        # for j in range(L+1):
        #     tmp += ((1 - (input-j).abs()).clamp(min=0))
        # # # print(tmp)

        # in_bounds = ((1 - (input-1).abs())).clamp(min=0,max=1)
        
        # if grad_input.min()<-1:
        #     print(grad_input.min(),grad_input.max())
        # lower_bound =  1
        # upper_bound = L
        
        # # Condition for m to be within the bounds
        # in_bounds = (lower_bound < input) & (input < upper_bound)
        # in_bounds = in_bounds.float()
        # in_bounds +=  (1-(1-input).abs())*(lower_bound >= input).float()*(input > 0).float()
        # in_bounds +=  (1-(input-L).abs())*(input >= upper_bound).float()*(upper_bound+1 > input).float()

        lower_bound = 0.5 #* S
        upper_bound = (L + 0.5) #* S
        
        # Condition for m to be within the bounds
        in_bounds = (lower_bound <= input) & (input <= upper_bound)
    
        
        # Convert boolean to -1 or 1
        gradient = in_bounds.float()
        grad_input = gradient*grad_input


        return grad_input, None, None