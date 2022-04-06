import torch
import torch.nn as nn
import scipy.io
import numpy as np

from printer.reconstruction_printer import reconstruction_printer
from models.DEQ import ReconstructionDEQ
from models.ForwardBackward import ForwardBackwardLayer
from solver.anderson import anderson

tomosipo = True
i = 200
path_model = f"model_weight.pth"
path_sino = f"data/test/input/{i}.mat"
path_img = f"data/test/output/{i}.mat"

model_infos = torch.load(path_model)
batch_size = model_infos['batch_size']
lr_model = model_infos['lr_model']
MAX_EPOCH = model_infos['MAX_EPOCH']
num_epochs = model_infos['num_epochs']
lr_fb = model_infos['lr_fb']
tol = model_infos['tol']
max_iter = model_infos['max_iter']
beta = model_infos['beta']
lam = model_infos['lam']
batch_size = model_infos['batch_size']

if tomosipo:
    from operators.radon import RadonTransform
    operator = RadonTransform(channels=1)
else:
    operator = torch.randn((110, 512))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

f = ForwardBackwardLayer(operator, lr_fb, tomosipo=tomosipo)
model = ReconstructionDEQ(f, anderson, tol=tol, max_iter=max_iter, beta=beta, lam=lam).to(device)
model.load_state_dict(model_infos['model_state_dict'])
model.eval()

sino = torch.from_numpy(np.array([[scipy.io.loadmat(path_sino)["data"]]])).to(device)

x_result = model(torch.zeros(1,1,512,512).to(device), sino)

if torch.is_tensor(operator):
    loss = nn.MSELoss()(operator @ x_result, sino)
else:
    loss = nn.MSELoss()(operator.torch_operator(x_result), sino).to(device)

loss = loss.item()


print("Loss for this image : "+str(loss))

img = torch.from_numpy(np.array([[scipy.io.loadmat(path_img)["data"]]])).to(device)
reconstruction_printer(x_result, img, i)