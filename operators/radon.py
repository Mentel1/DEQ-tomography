import torch
import tomosipo as ts

from tomosipo.torch_support import to_autograd

class RadonTransform():
    '''
    A class that creates a 'tomographer', an object that performs a radon
    transform on a volume.

    The purpose of this project is to reconstruct images from sinograms.
    Those sinograms were obtained by radon transform over the former images.
    We want to incorporate this operator in our pipeline in order to properly solve
    the inverse problem (that is, our image reconstruction problem).
    '''

    def __init__(self, channels=1, image_size=(512, 512), nb_angles=110) -> None:
        vg = ts.volume(shape=(channels, *image_size))
        pg = ts.parallel(angles=nb_angles, shape=(channels, image_size[1]))
        self.operator = ts.operator(vg, pg)
        self.torch_operator = to_autograd(self.operator)
        self.transposed_torch_operator_ = to_autograd(self.operator.T)
        
    def T(self, x):
        '''
        Transposed operator apply
        '''
        return self.transposed_torch_operator_(x)

    def __call__(self, x):
        '''
        Redifines calling the operator on a tensor
        '''
        return self.torch_operator(x)