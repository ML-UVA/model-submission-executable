import torch

from firebase import Firebase

def load_dataloader(func_id):
    # url = competition['url']
    # output = 'dataloader.pt'
    # download(url, output, quiet=True)
    dataloader = torch.load(f'dataloaders/dataloader_{func_id}.pt', weights_only=False)
    # remove('dataloader.pt')
    return dataloader