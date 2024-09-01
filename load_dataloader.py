# Customizable function that will return a dataloader specific to the competition
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_dataloader(competition):
    if competition == 1:
        def f(x):
            return x ** 2 + 2 * x + 1 # x^2 + 2x + 1
    elif competition == 2:
        def f(x):
            return -5 * (x ** 2) + 3.2 * x - 4 # -5x^2 + 3.2x - 4
    
    x = np.random.rand(1000) * 100
    y = np.array([f(i) for i in x])

    tensor_x = Tensor(x).reshape((len(x), 1))
    tensor_y = Tensor(y).reshape((len(y), 1))

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader
