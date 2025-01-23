import torch
from torch.autograd import Function

class GZIF(Function):
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

        # Efficiently compute the gradient using a single expression
        gradient = (input * (input < 1.0).float()) + (1.0 * ((input >= 1.0) & (input <= L)).float()) + ((1 - (input - L)) * (input > L).float())
        gradient = torch.clamp(gradient, min=0, max=1)
        grad_input = gradient*grad_input
        return grad_input, None, None