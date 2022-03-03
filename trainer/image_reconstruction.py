import torch.nn as nn

def reconstruction_epoch(loader, model, operator, device, opt=None, lr_scheduler=None):
    """
    The trainer to call at each epoch
    """
    total_loss = 0.
    _ = model.eval() if opt is None else model.train()
    for radon_trans, images in loader:
        images, radon_trans = images.to(device), radon_trans.to(device)
        x_inf = model(images, radon_trans)
        loss = nn.MSELoss()(operator @ x_inf, radon_trans)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_loss += loss.item() * images.shape[0]

    return total_loss / len(loader.dataset)