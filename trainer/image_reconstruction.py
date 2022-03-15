import torch.nn as nn

def reconstruction_epoch(loader, model, operator, device, opt=None, lr_scheduler=None):
    """
    The trainer to call at each epoch
    """
    total_loss = 0.
    _ = model.eval() if opt is None else model.train()
    for sinograms, images in loader:
        images, sinograms = images.to(device), sinograms.to(device)
        x_inf = model(images, sinograms)

        loss = nn.MSELoss()(x_inf, images)
        
        # if type(operator) == 'Tensor':
        #   loss = nn.MSELoss()(operator @ x_inf, sinograms)
        # else:
        #   loss = nn.MSELoss()(operator(x_inf), sinograms)
          
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_loss += loss.item() * images.shape[0]

    return total_loss / len(loader.dataset)