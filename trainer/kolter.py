import torch.nn as nn

def epoch(loader, model, device, opt=None, lr_scheduler=None):
    """
    The trainer to call at each epoch
    """
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.MSELoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_loss += loss.item() * X.shape[0]
        
    return total_loss / len(loader.dataset)