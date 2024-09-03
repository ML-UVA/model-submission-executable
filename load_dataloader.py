from torch import load
from gdown import download
from os import remove

def load_dataloader(competition):
    url = competition['url']
    output = 'dataloader.pt'
    download(url, output, quiet=True)
    dataloader = load('dataloader.pt')
    remove('dataloader.pt')
    return dataloader