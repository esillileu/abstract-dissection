from torch.autograd import Function


class relu(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        output_tensor = input_tensor.clamp(min=0)
        ctx.save_for_backward(input_tensor)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        return grad_output * (input_tensor > 0).float()
