import os
from torchvision.utils import save_image  # Speichern von Bildern
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_metrics_loss(modelname,losses):
    path = "./data/outputs/"
    os.makedirs(path, exist_ok=True)
    EPOCH_COUNT= range(1,len(losses)+1) # Anzahl der Epochen vom Dis.
    fig = plt.figure(figsize=(10,5))
    plt.title("Loss während dem Training")
    plt.plot(EPOCH_COUNT,losses,"b-", label="Loss")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend()
    name = "loss"+modelname+".png"
    fig.savefig(path+name, dpi=fig.dpi)

def plot_metrics_acc(modelname,losses):
    path = "./data/outputs/"
    os.makedirs(path, exist_ok=True)
    EPOCH_COUNT= range(1,len(losses)+1) # Anzahl der Epochen vom Dis.
    fig = plt.figure(figsize=(10,5))
    plt.title("Acc während dem Training")
    plt.plot(EPOCH_COUNT,losses,"b-", label="Acc")
    plt.xlabel("EPOCH")
    plt.ylabel("Acc")
    plt.legend()
    name = "acc"+modelname+".png"
    fig.savefig(path+name, dpi=fig.dpi)


def show_images(e, x, x_adv, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1, 2, 0)))
        axes[0, i].set_title("Normal")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].set_title("Adv")

    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))
