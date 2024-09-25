from load_dataloader import *

import torch

class Eval:
    def __init__(self, model, dataloader):
        self.dataloader = dataloader
        print('Loaded dataloader')
        self.model = torch.jit.load(model)
        print('Loaded model')

    def eval(self, metrics):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                nan = torch.isnan(labels)
                inputs = torch.Tensor([inputs[i] for i in range(len(inputs)) if nan[i] == 0])
                labels = torch.Tensor([labels[i] for i in range(len(labels)) if nan[i] == 0])
                inputs = inputs[:, None]
                outputs = self.model(inputs).squeeze()

                all_preds += list(outputs.cpu().numpy())
                all_labels += list(labels.cpu().numpy())
        # print(all_labels)
        # print(all_preds)
        res = {name: float(metric(all_labels, all_preds)) for name, metric in metrics}
        return res

