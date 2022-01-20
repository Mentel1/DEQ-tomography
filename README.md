# Introduction

[TO COME]

# Project Architecture

## Data

We obviously didn't push our dataset to Github for performance and confidentiality issues. However, there are a few **requirements** when it comes to the dataset folders organization. Our _data loaders_ expect a specific architecture:

```bash
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
```

where each input and output folder couples should have the same number of files. Furthermore, for a given input file, the corresponding output file should have the same name (i.e if the input file is `13.mat`, so is the ouput file in the output folder).

Finally, you should change a few lines of code in `utils.DataLoaders.py` depending on the lengths of your datasets, right here:

```python
    def __len__(self):
        if self.set_name_ == "train":
            return 21  # The size of the train dataset
        elif self.set_name_ == "test":
            return 10  # The size of the test dataset
        else:
            return 10  # The size of the validationd dataset
```

Feel free to modify the `utils.DataLoaders.py` even further, to better suit your needs and your input data.

# To do

- [x] Design the model's architecture

- [x] Choose a loss function specific to the problem we're trying to solve

- [x] Implement a data loading class to plug at the very beginning of any pytorch pipeline.

- [ ] Create the actual pytorch model, according to the design we agreed upon.

- [ ] Create the trainers

- [ ] Create the monitors to keep track of the performances throughout training and after.
