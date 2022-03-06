import tomosipo as ts

class RadonTransform():
    '''
    A class that creates a 'tomographer', an object that performs a radon
    transform on a volume.

    The purpose of this project is to reconstruct images from sinograms.
    Those sinograms were obtained by radon transform over the former images.
    We want to incorporate this operator in our pipeline in order to properly solve
    the inverse problem (that is, our image reconstruction problem).
    '''

    def __init__(self, image_batch_size=5, image_size=(512, 512), nb_angles=110) -> None:
        vg = ts.volume(shape=(image_batch_size, *image_size))
        pg = ts.parallel(angles=nb_angles, shape=(image_batch_size, image_size[1]))
        self.operator_ = ts.operator(vg, pg)

    def operator(self):
        '''
        Returns the instantiated operator
        '''
        return self.operator_