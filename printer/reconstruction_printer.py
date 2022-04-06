import matplotlib.pyplot as plt
import numpy as np


def reconstruction_printer(x_result, img, i):

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(np.flipud(x_result.detach().cpu().numpy().reshape(512, 512)), cmap='gray')
    axes[1].imshow(img.detach().cpu().numpy().reshape(512, 512), cmap='gray')
    # Un-comment line if you work on a GUI-backend 
    # plt.show()
    plt.savefig(f"results_{i}.jpg")