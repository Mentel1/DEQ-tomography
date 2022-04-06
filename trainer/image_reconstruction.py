import torch.nn as nn
import torch

def reconstruction_epoch(loader, model, operator, device, opt=None, lr_scheduler=None):
    """
    The trainer to call at each epoch
    """
    total_loss = 0.
    N = 0
    _ = model.eval() if opt is None else model.train()
    for sinograms, images in loader:
        images, sinograms = images.to(device), sinograms.to(device)
        x_inf = model(images, sinograms)

        if torch.is_tensor(operator):
          loss = nn.MSELoss()(operator @ x_inf, sinograms)
        else:
          loss = nn.MSELoss(reduction='mean')(operator.torch_operator(x_inf), sinograms).to(device)
          
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_loss += loss.item()
        N += 1

    return total_loss / N