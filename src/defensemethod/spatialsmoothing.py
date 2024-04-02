import torch
import torchvision
from scipy import ndimage
from tqdm import tqdm
import torchattacks

import sys
sys.path.append('./src')
sys.path.append('./src/metrics')
sys.path.append('./src/metrics/plot/')
sys.path.append('./src/metrics/acc/')


from plot import plot_metrics_acc_batch, plot_metrics_loss_batch, show_images


def gaussianblur(images):
    #for i , (images, labels) in enumerate(testloader,0):
    transform = torchvision.transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 5.))
    blur_img = transform(images)
    return blur_img

def median_smoothing(testloader,kernel_size=2):
    # Smooth image by apllying median filter with kernel size (m*n). Use default kernel size of 2x2 as indicated by CW,
    '''
        Try to calculate median of given pixels inside of kernel window to normalize it's values.
        The function can be used with batch or single image array by measuring its shape and thereby it's dimension.
        The image data is casted into torch sensor because the scipy filter will change it's type otherwise to ndarray format.
        Usage for batch: smoothed_batch = median_smoothing(original_batch)
        Usage for single image: smoothed_image = median_smoothing(original_batch[index_of_image][0])
    '''
    for i , (images, labels) in enumerate(testloader,0):
        img_data_median = torch.clone(images) #Clone tensor to protect original tensor data
        for j in range(img_data_median.shape[0]):
            for k in range(img_data_median.shape[1]): #RGB Values
                    img_data_median[j][k] = torch.from_numpy(ndimage.median_filter(img_data_median[j][k], size=kernel_size)) #Use ndimage filter for CIFAR-10
    return(img_data_median)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def spatialsmoothingTest(model_path,  testloader, device, model, modelname):
    torch.cuda.empty_cache()

    testloader = testloader
    model_point = torch.load(model_path, map_location=device)
    model = model
    model.load_state_dict(model_point["state_dict"])
    val_loss=0
    val_accuracy=0
    
    loss_per_batch = []
    acc_per_batch = []
    
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    model.eval()
    i =0
    for images, labels in tqdm(testloader):

        images, labels = images.to(device), labels.to(device)

        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        adv_images = attack(images, labels)
        smooth_images = gaussianblur(adv_images)
        with torch.no_grad():
        #predictions = model(images).to(device)
            predictions_adv = model(smooth_images).to(device)
        show_images(i, images, smooth_images, "./data/outputs/smooth/")
        show_images(i+1, adv_images, smooth_images, "./data/outputs/smooth/")

        val_acc_batch = accuracy_fn(y_true=labels, y_pred=predictions_adv.argmax(dim=1))
        val_loss_batch = loss_func(predictions_adv, labels)

        
        loss_per_batch.append(val_loss_batch.item())
        plot_metrics_loss_batch(modelname, loss_per_batch, True)
        acc_per_batch.append(val_acc_batch)
        plot_metrics_acc_batch(modelname, acc_per_batch, True)
        
        val_loss += loss_func(predictions_adv, labels)
        #val_acc_batch = torch.eq(labels, predictions_adv.argmax(dim=1)).sum().item()
        #print("Batch",i,":" ,val_acc_batch)
        val_accuracy += accuracy_fn(y_true=labels, y_pred=predictions_adv.argmax(dim=1))
    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    print( "Loss: ", val_loss, "Acc: ",val_accuracy)

    Path_checkpoint = "./model/metrics/"+modelname+"_test_Checkpoint.pth"

    checkpoint = {
            "Loss": val_loss,
            "Acc": val_accuracy,
        }
    torch.save(checkpoint, Path_checkpoint)
    print("Loss: ", val_loss, "Acc.: ", val_accuracy)
