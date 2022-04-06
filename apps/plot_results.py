
import sys
sys.path.insert(0, 'C:/Users/Lila/Documents/Centrale/3A/Projet Deep Equilibrium/Git projet/DEQ-tomography')

import torch
import torch.nn as nn
from models.DEQ import ReconstructionDEQ
from solver.anderson import anderson
from models.ForwardBackward import ForwardBackwardLayer
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def eval(path_model, path_img, path_sino, tomosipo=True, file="result.jpg"):
    """
    Function plotting one image reconstructed for the sinogramme at path_sino, alongside the real image corresponding
    """

    if tomosipo:
        from operators.radon import RadonTransform
        operator = RadonTransform(channels=1)
    else:
        operator = torch.randn((110, 512))

    # f = ForwardBackwardLayer(operator, .01, tomosipo=tomosipo)
    # tol=1e-2
    # max_iter=50
    # beta=0.1
    # lam=1e-2
    # lr=1e-4
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Play with anderson's hyperparameter in case solver raises singular matrix errors
    # model = ReconstructionDEQ(f, anderson, tol=tol, max_iter=max_iter, beta=beta, lam=lam).to(device)
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
        loss = loss/1

    loss = loss.item()


    print("Loss for this image : "+str(loss))

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(np.flipud(x_result.detach().cpu().numpy().reshape(512, 512)), cmap='gray')
    axes[1].imshow(img.detach().cpu().numpy().reshape(512, 512), cmap='gray')
    plt.show()
    plt.savefig(file)

if __name__ == "__main__":

    i = 200
    eval(f"model_weights.pth", f"data/test/output/{i}.mat", f"data/test/input/{i}.mat", tomosipo=True, file=f"results_{i}.jpg")