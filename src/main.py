#Import runtime libraries
import os
import sys

# Append needed function/module paths
sys.path.append('./src')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/new_resnet')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/ResNet_train')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/dataloadernew')

sys.path.append('./src')
sys.path.append('./src/metrics/')
sys.path.append('./src/metrics/acc')

sys.path.append('./src/training')
sys.path.append('./src/training/train')

sys.path.append('./src/testing')
sys.path.append('./src/testing/testing')
sys.path.append('./src/testing/attacks')


sys.path.append('./src/utils')
sys.path.append('./src/utils/utils')

sys.path.append('./src')
sys.path.append('./src/defensemethod/')
sys.path.append('./src/defensemethod/adversarialtraining')
sys.path.append('./src/defensemethod/spatialsmoothing')
sys.path.append('./src/defensemethod/genadvexamples')

sys.path.append('./src/utils/utils')
sys.path.append('./src/utils/')
sys.path.append('./src/utils/config')

#custom moduls
from utils import get_device
from new_resnet import ResNet50, ResNet18
from train import train
import dataloadernew
from testing import test
from attacks import adversarialattack
from adversarialtraining import AdversarialTraining
from spatialsmoothing import spatialsmoothingTest
from genadvexamples import gen_adv
import utils
from config import config

#packages
import torch
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(config.seed)

#Support function for clearing terminal output
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

#Project main
def project_main (device, resnet18, resnet50):

    print("##########")
    print("Abwehr von State-of-the-Art Black-Box Adversarial Attacks auf Audio Deepfake Detektionmodelle mittels Adversarial Training und Spatial Smoothing​")
    print("##########")

    print("\n")
    print("Models")
    print("0. All Models (Training & Testing)")
    print("1. ResNet18 (Baseline)")
    print("2. ResNet50 (Baseline) ")
    print("3. Adversarial Attack on ResNet18")
    print("4. Adversarial Attack on ResNet50")
    print("5. Adversarial Attack on ResNet18 with Adversarial Training")
    print("6. Adversarial Attack on ResNet18 with Spatial Smoothing")
    print("7. Adversarial Attack on ResNet50 with Adversarial Training")
    print("8. Adversarial Attack on ResNet50 with Spatial Smoothing")
    print("##########")
    print("9. Call Checkpoints (Metrics - Losses and Acc) of all Models")
    print("##########")

    user_input = int(input("Model:"))
    if user_input == 0:
        cls()
        #call ResNet18 and ResNet50

        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        
        print("\n")
        print("### Training ResNet18 ###")
        path_resnet18 = "./model/resnet18.pth"
        modelname_resnet18 = "_resnet18_"
        train(resnet18, train_batches, device, path_resnet18, modelname_resnet18)
        
        print("\n")
        print("### Training Resnet50 ###")
        path_resnet50 = "./model/resnet50.pth"
        modelname_resnet50 = "_resnet50_"
        train(resnet50, train_batches, device, path_resnet50, modelname_resnet50)
        
        print("\n")
        print("### Testing ResNet18 ###")
        print("### Normal Testing on ResNet18 ###")
        modelname_resnet18_test = "_resnet18_test_"
        test(resnet18, val_batches, device, path_resnet18, modelname_resnet18_test)
        
        print("\n")
        print("### Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_aa = "_resnet18_test_aa_"
        adversarialattack(path_resnet18, val_batches, device, resnet18, modelname_resnet18_test_aa)

        print("\n")
        print("### Testing ResNet50 ###")
        print("### Normal Testing on ResNet50 ###")
        modelname_resnet50_test = "_resnet50_test_"
        test(resnet50, val_batches, device, path_resnet50, modelname_resnet50_test)
        
        print("\n")
        print("### Adversarial Attack on ResNet50 ###")
        modelname_resnet50_test_aa = "_resnet50_test_aa_"
        adversarialattack(path_resnet50, val_batches, device, resnet50, modelname_resnet50_test_aa)

        # Defense Method
        print("\n")
        print("### Spatial Smoothing ###")
        print("### ResNet18 ###")
        modelname_resne18_test_smoothing = "_resnet18_smoothing_"
        spatialsmoothingTest(path_resnet18, val_batches, device, resnet18,modelname_resne18_test_smoothing)

        print("\n")
        print("### ResNet50 ###")
        modelname_resne50_test_smoothing = "_resnet50_smoothing_"

        spatialsmoothingTest(path_resnet50, val_batches, device, resnet50, modelname_resne50_test_smoothing)

        print("\n")
        print("### Adversarial Training ###")
        filename_resnet18 = "./data/data_advtrain/datar18.tar"
        filename_resnet50 = "./data/data_advtrain/datar50.tar"
        
        print("### ResNet18 ###")
        path_resnet18_adv_train = "./model/resnet18_adv_train.pth"
        modelname_resnet18_adv_train = "_resnet18_adv_train_"
        AdversarialTraining(resnet18, train_batches, device, path_resnet18_adv_train, modelname_resnet18_adv_train, filename_resnet18)

        print("\n")
        print("### ResNet50 ###")
        path_resnet50_adv_train = "./model/resnet50_adv_train.pth"
        modelname_resnet50_adv_train = "_resnet50_adv_train_"
        AdversarialTraining(resnet50, train_batches, device, path_resnet50_adv_train, modelname_resnet50_adv_train, filename_resnet50)

        print("\n")
        print("### Testing ###")
        print("### Normal Testing on ResNet18 ###")
        modelname_resnet18_at_test = "_resnet18_at_test_"
        test(resnet18, val_batches, device, path_resnet18_adv_train, modelname_resnet18_at_test)

        print("\n")
        print("### Testing with Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_at_aa = "_resnet18_test_at_aa_"
        adversarialattack(path_resnet18_adv_train, val_batches, device, resnet18, modelname_resnet18_test_at_aa)
        
        print("\n")
        print("### Normal Testing on ResNet50 ###")
        modelname_resnet50_at_test = "_resnet50_at_test_"
        test(resnet50, val_batches, device, path_resnet50_adv_train, modelname_resnet50_at_test)

        print("\n")
        print("### Testing with Adversarial Attack on ResNet50 ###")
        modelname_resnet50_test_at_aa = "_resnet50_test_at_aa_"
        adversarialattack(path_resnet50_adv_train, val_batches, device, resnet50, modelname_resnet50_test_at_aa)

    elif user_input == 1:
        cls()

    elif user_input == 2:
        cls()

    elif user_input == 3:
        cls()

    elif user_input==4:
        cls()

    elif user_input == 5:
        cls()

    elif user_input == 6:
        cls()

    elif user_input == 7:
        cls()
        print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")
        print("### ResNet18 ###")
        modelname_resnet18 = "_resnet18_"

        output = torch.load("./model/metrics/"+modelname_resnet18+"Checkpoint.pth")
        print(output)

        print("### ResNet50 ###")
        modelname_resnet50 = "_resnet50_"
        output = torch.load("./model/metrics/"+modelname_resnet50+"Checkpoint.pth")
        print(output)
        
        print("### ResNet18 with Adversarial Training ###")
        modelname_resnet18_adv_train = "_resnet18_adv_train_"
        output = torch.load("./model/metrics/"+modelname_resnet18_adv_train+"Checkpoint.pth")
        print(output)
    
    
        print("### ResNet50 with Adversarial Training ###")
        modelname_resnet50_adv_train = "_resnet50_adv_train_"
        output = torch.load("./model/metrics/"+modelname_resnet50_adv_train+"Checkpoint.pth")
        print(output)

device = get_device()
resnet18 = ResNet18().to(device)
resnet50 = ResNet50().to(device)
print(resnet50)
project_main(device, resnet18, resnet50)
