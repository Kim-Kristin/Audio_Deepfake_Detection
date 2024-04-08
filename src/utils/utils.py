import torch


def get_device():
    if torch.backends.mps.is_available():
      return torch.device("mps") # macos
    if torch.cuda.is_available():
        return torch.device('cuda') #nvidia
    else:                         
        return torch.device('cpu')

class DeviceDataLoader():

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    def __len__(self):
        return len(self.dataloader)
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)