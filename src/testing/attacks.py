
#https://foolbox.jonasrauber.de/guide/examples.html
import torch
import torchattacks
from tqdm import tqdm

# Calculates accuracy between truth labels and predictions.
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def adversarialattack(model_path, testloader, device, model, modelname):
    torch.cuda.empty_cache()

    testloader = testloader
    model_point = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.load_state_dict(model_point["state_dict"])
    val_loss=0
    val_accuracy=0

    loss_test = []
    acc_test = []

    i = 1
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    for i, (images, labels) in tqdm(enumerate(testloader,0)):

        images, labels = images.to(device), labels.to(device)

        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        adv_images = attack(images, labels)

        with torch.no_grad():
            predictions_adv = model(adv_images).to(device)
        val_loss += loss_func(predictions_adv, labels)
        val_acc_batch = torch.eq(labels, predictions_adv.argmax(dim=1)).sum().item()
        print("Batch",i,":" ,val_acc_batch)
        val_accuracy += accuracy_fn(y_true=labels, y_pred=predictions_adv.argmax(dim=1))
        i+=1
    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    print( "Loss: ", val_loss, "Acc: ",val_accuracy)
    Path_checkpoint = "./model/metrics/"+modelname+"test_with_attack_Checkpoint.pth"

    checkpoint = {
        "Loss": loss_test,
        "Acc": acc_test,
    }
    torch.save(checkpoint, Path_checkpoint)
    return {"Loss: ":val_loss.item(), "Acc.: ": val_accuracy}
