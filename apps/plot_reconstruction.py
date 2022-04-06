import torch
import torch.nn as nn
from models.DEQ import ReconstructionDEQ
from solver.anderson import anderson
from models.ForwardBackward import ForwardBackwardLayer
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model, map_location=device)
model.eval()
img = torch.from_numpy(np.array([[scipy.io.loadmat(path_img)["data"]]])).to(device)
# The following line enables to test that the model doesn't use the image data to compute an image, only the sinogramm
# img = torch.zeros_like(img)
sino = torch.from_numpy(np.array([[scipy.io.loadmat(path_sino)["data"]]])).to(device)

x_result = model(img, sino)

if torch.is_tensor(operator):
    loss = nn.MSELoss()(operator @ x_result, sino)
else:
    loss = nn.MSELoss()(operator.torch_operator(x_result), sino).to(device)

loss = loss.item()


print("Loss for this image : "+str(loss))

fig, axes = plt.subplots(1,2)
axes[0].imshow(np.flipud(x_result.detach().cpu().numpy().reshape(512, 512)), cmap='gray')
axes[1].imshow(img.detach().cpu().numpy().reshape(512, 512), cmap='gray')
plt.show()
plt.savefig(f"results_{i}.jpg")