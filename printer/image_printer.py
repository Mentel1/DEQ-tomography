import torch
import matplotlib.pyplot as plt

def print_grayscale(loader, image_index, input_image=True):
    """
    A function that prints a grayscale image of our liking in a data set of gray images
    """
    dataiter = iter(loader)

    if input_image:
        images, _ = next(dataiter)
    else:
        _, images = next(dataiter)

    gray_image = images[image_index].numpy()
    plt.imshow(gray_image, cmap='gray')
    plt.show()