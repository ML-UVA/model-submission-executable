from load_dataloader import *

from torch.jit import load
from torch import device, cuda, no_grad

class Eval:
    def __init__(self, path, competition):
        self.dataloader = load_dataloader(competition)
        print('Loaded dataloader')
        self.model = load(path)
        print('Loaded model')

    def eval(self, metrics):
        all_preds = []
        all_labels = []

        with no_grad():
            for inputs, labels in self.dataloader:
                outputs = self.model(inputs).squeeze()

                all_preds += list(outputs.cpu().numpy())
                all_labels += list(labels.cpu().numpy())

        res = {name: float(metric(all_labels, all_preds)) for name, metric in metrics}
        return res

