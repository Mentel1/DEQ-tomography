import torch
import torch.nn as nn
import torch.autograd as autograd

class DEQFixedPoint(nn.Module):
    """
    Class for fixed point deep equilibrium
    """
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        self.backward_res, self.forward_res = None, None
        
    def forward(self, x):
        """
        compute forward pass and re-engage autograd tape
        """
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            """
            A hook needed during the backward pass
            """
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z

class ReconstructionDEQ(nn.Module):
    """
    Class for fixed point deep equilibrium
    """
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        self.backward_res, self.forward_res = None, None
        
    def forward(self, input_batch, targets):
        """
        compute forward pass and re-engage autograd tape
        """
        with torch.no_grad():
            fixed_point, self.forward_res = self.solver(lambda x: self.f(x, targets), torch.zeros_like(input_batch), **self.kwargs)

        z = self.f(fixed_point, targets)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, targets)

        def backward_hook(grad):
            """
            A hook needed during the backward pass
            """
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z
        