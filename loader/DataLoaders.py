import torch
import scipy.io
import os
import numpy as np


class Tomography(torch.utils.data.Dataset):
    '''
    A data loader designed to be plugged into a pytorch DL pipeline.
    '''

    def __init__(self, root, set_name="train", transform=False):
        '''
        The class initializer.

        Parameters
        ----------

        root: str
            The root of the directory that contains all your datasets according to the following architecture:
                .data/
                    train/
                        input/
                            1.mat
                            2.mat
                            ...
                        output/
                            1.mat
                            2.mat
                            ...
                    test/
                        input/
                            1.mat
                            2.mat
                            ...
                        output/
                            1.mat
                            2.mat
                            ...
                    valid/
                        input/
                            1.mat
                            2.mat
                            ...
                        output/
                            1.mat
                            2.mat
                            ...
            Each input and output folder couples should have the same number of files and for a given input file, the corresponding output file should have the same name (if input file is "13.mat", so is the ouput file in the output folder).

        set_name: str
            It can be "train", "test" or "valid" and defaults to the first one. It indicates which dataset to load (either for training, for testing or for validation).

        transform: bool or torchvision transformation (e.g torchvision.transforms.Grayscale)
            It defaults to False, indicating that no transformation whatsoever should be applied. Otherwise, if a transformation function is passed as an argument, it will be applied to the data.

        '''

        # Make sure the set_name is valid
        assert set_name in [
            "train", "test", "valid"], "The set name should be 'train', 'test' or 'valid."
        self.set_name_ = set_name

        # Instantiate the filepath
        if set_name == "train":
            filepath = f"{root}train/"
        elif set_name == "test":
            filepath = f"{root}test/"
        else:
            filepath = f"{root}valid/"

        input_path = f"{filepath}input"
        output_path = f"{filepath}output"

        # We open and store the input data
        inputs_list = []
        outputs_list = []
        for filename in os.listdir(input_path):
            # Scipy loads an object and the "data" key of the object stores the actual data in a numpy array
            input_data = scipy.io.loadmat(f"{input_path}/{filename}")["data"]
            inputs_list.append([input_data])
            output_data = scipy.io.loadmat(f"{output_path}/{filename}")["data"]
            outputs_list.append([output_data])

        # Put the data into tensors
        # Unsqueeze to add the channel dimension (shape 1 for grayscale)
        self.inputs_ = torch.from_numpy(np.concatenate(
            inputs_list, axis=0)).unsqueeze(1)
        self.outputs_ = torch.from_numpy(np.concatenate(
            outputs_list, axis=0)).unsqueeze(1)

        # Apply transformations if need be
        if transform:
            self.inputs_ = transform.forward(self.inputs_)

    def __getitem__(self, index):
        return self.inputs_[index], self.outputs_[index]

    def __len__(self):
        if self.set_name_ == "train":
            return 2500  # The size of the train dataset
        elif self.set_name_ == "test":
            return 500  # The size of the test dataset
        else:
            return 10  # The size of the validationd dataset
