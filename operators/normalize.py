from torch import is_tensor, max, min
import numpy as np

def normalize(x):

    if is_tensor(x):
        max_value = max(x)
        min_value = min(x)
        return (x-min_value)/(max_value-min_value)
    else:
        max_value = np.max(x)
        min_value = np.min(x)
        return (x-min_value)/(max_value-min_value)
