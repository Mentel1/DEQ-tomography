import torch
import torch.nn as nn
import scipy.io
import numpy as np

from printer.reconstruction_printer import reconstruction_printer

tomosipo = True
i = 200
path_model = f"model_weights.pth"
path_sino = f"data/test/input/{i}.mat"
path_img = f"data/test/output/{i}.mat"

if tomosipo:
    from operators.radon import RadonTransform
    operator = RadonTransform(channels=1)
else:
    operator = torch.randn((110, 512))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model, map_location=device)
model.eval()

sino = torch.from_numpy(np.array([[scipy.io.loadmat(path_sino)["data"]]])).to(device)

x_result = model(torch.zeros(512,512), sino)

if torch.is_tensor(operator):
    loss = nn.MSELoss()(operator @ x_result, sino)
else:
    loss = nn.MSELoss()(operator.torch_operator(x_result), sino).to(device)

loss = loss.item()


print("Loss for this image : "+str(loss))

img = torch.from_numpy(np.array([[scipy.io.loadmat(path_img)["data"]]])).to(device)
reconstruction_printer(x_result, img)