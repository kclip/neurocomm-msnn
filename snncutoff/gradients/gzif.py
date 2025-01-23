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
        L = torch.tensor(L)
        out = input.floor().clamp(0.0,L)
        ctx.save_for_backward(input, out,L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out,L) = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        lower_bound = 0.5 
        upper_bound = (L + 0.5) 
        # Condition for m to be within the bounds
        # in_bounds = (lower_bound <= input) & (input <= upper_bound)
        # gradient = in_bounds.float()

        # Efficiently compute the gradient using a single expression
        gradient = (input * (input < 1.0).float()) + (1.0 * ((input >= 1.0) & (input <= L)).float()) + ((1 - (input - L)) * (input > L).float())
        gradient = torch.clamp(gradient, min=0, max=1)
        grad_input = gradient*grad_input
        return grad_input, None, None