import torch


def get_device():
    if torch.backends.mps.is_available():
      return torch.device("mps")
    if torch.cuda.is_available():     # Wenn cuda verfügbar dann:
        return torch.device('cuda')   # Nutze Device = Cuda (=GPU)
    else:                         # Ansonsten
        return torch.device('cpu')    # Nutze Device = CPU


class DeviceDataLoader():

    # Initialisierung
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    # Anzahl der Images pro Batch
    def __len__(self):
        return len(self.dataloader)

    # Erstellt einen Batch an Tensoren nach dem Verschieben auf das Device
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)


#device = get_device()
#print(device)
