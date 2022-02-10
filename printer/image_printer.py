import torch
import matplotlib.pyplot as plt

def print_grayscale(loader, image_index):
    """
    A function that prints a grayscale image of our liking in a data set of gray images
    """
    dataiter = iter(loader)
    images, _ = next(dataiter)

    gray_image = images[image_index].to(torch.uint8)[0, :, :]
    npimg = gray_image.numpy()
    plt.imshow(npimg, cmap='gray', vmin=0, vmax=255)
    plt.show();