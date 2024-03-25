import torch
import torchattacks
from tqdm import tqdm
from torch.utils.data import TensorDataset

import sys
sys.path.append('./src')
sys.path.append('./src/metrics')
sys.path.append('./src/metrics/plot/')
sys.path.append('./src/metrics/acc/')

from plot import plot_metrics_loss, plot_metrics_acc, show_images
import acc

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def AdversarialTraining(model, trainloader, device, path, modelname, filename):
    epochs = 10
    LEARNING_RATE = 0.001
    GRADIENT_MOMENTUM = 0.90
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM)

    loss_per_epoch = []
    acc_per_epoch = []
    batch_size = 16

    # load normal and adv data
    train_data = torch.load(filename, map_location=device)
    print(len(train_data["adv"]))
    #train_dataset = TensorDataset(train_data["normal"],train_data["adv"]) #rain_data["label_normal"]) # train_data["adv"], train_data["label_normal"], train_data["label_normal"])
    #train_label = train_data["label_normal"]
    #train_data = TensorDataset(train_data, train_label)
    #x_tmp = train_data["normal"][:5]
    #adv_tmp = train_data["adv"][:5]
    #train_loader= torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #train_loader_label = torch.utils.data.DataLoader(train_label)
    model.train()

    for epoch in range(epochs):
        train_loss_normal, train_loss_adv = 0, 0
        train_loss, train_acc_epoch, train_acc = 0, 0, 0
        correct_adv, correct_normal, correct_total, acc = 0, 0, 0, 0
        #show_images(epoch, x_tmp, adv_tmp, "./data/outputs/Images/")


        for i, data in tqdm(enumerate(train_data,0)):
            normal, adv, normal_labels = data[0], data[1], data[2]

            predictions_normal = model(normal).to(device)
            loss_normal = loss_func(predictions_normal, normal_labels)
            train_loss_normal += loss

            predictions_adv = model(adv).to(device)
            loss_adv = loss_func(predictions_adv, normal_labels)
            train_loss_adv += loss

            loss = loss_normal + loss_adv
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #acc
            correct_normal = torch.eq(normal_labels, predictions_normal).sum().item()
            correct_adv = torch.eq(normal_labels, predictions_adv).sum().item()
            correct_total = correct_adv + correct_normal
            train_acc = (correct_total / (len(normal_labels)+ len(normal_labels))) * 100

        train_loss /= (len(normal_labels)+ len(normal_labels))
        #train_acc_epoch = (correct/ len(label_sum)) * 100
        train_acc /= (len(normal_labels)+ len(normal_labels))
            # Save losses & plot
        loss_per_epoch.append(train_loss.item())
        plot_metrics_loss(modelname, loss_per_epoch)
        acc_per_epoch.append(train_acc)
        plot_metrics_acc(modelname, acc_per_epoch)

        print( "Epoch", epoch+1,"/",epochs, "Loss: ",
                    train_loss, "Acc: ",train_acc)

        Path_checkpoint = "./model/metrics/"+modelname+"Checkpoint.pth"

        checkpoint = {
            "Loss": loss_per_epoch,
            "Acc": acc_per_epoch,
        }
        torch.save(checkpoint, Path_checkpoint)

    torch.save({"state_dict": model.state_dict(),"Loss":train_loss.item(), "Acc.": train_acc}, path)
    return {"Loss: ":train_loss.item(), "Acc.: ": acc}
