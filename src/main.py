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
    print("7. Call Checkpoints (Metrics - Losses and FID) of all Models")
    print("##########")

    user_input = int(input("Model:"))
    if user_input == 0:
        cls()
        # call ResNet18 and ResNet50

        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)

        print("### Training ResNet18 ###")
        path_resnet18 = "./model/resnet18.pth"
        modelname_resnet18 = "_resnet18_"
        #train(resnet18, train_batches, device, path_resnet18, modelname_resnet18)

        print("### Training Resnet50 ###")
        path_resnet50 = "./model/resnet50.pth"
        modelname_resnet50 = "_resnet50_"
        #train(resnet50, train_batches, device, path_resnet50, modelname_resnet50)

        print("### Testing ResNet18 ###")
        print("### Normal Testing on ResNet18 ###")
        modelname_resnet18_test = "_resnet18_test_"

        #test(resnet18, val_batches, device, path_resnet18, modelname_resnet18_test)

        print("### Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_aa = "_resnet18_test_aa_"
        #adversarialattack(path_resnet18, val_batches, device, resnet18, modelname_resnet18_test_aa)

        print("### Testing ResNet50 ###")
        print("### Normal Testing on ResNet50 ###")

        modelname_resnet50_test = "_resnet50_test_"
        #test(resnet50, val_batches, device, path_resnet50, modelname_resnet50_test)

        print("### Adversarial Attack on ResNet50 ###")

        modelname_resnet50_test_aa = "_resnet50_test_aa_"
        #adversarialattack(path_resnet50, val_batches, device, resnet50, modelname_resnet50_test_aa)


        # Defense Method
        print("### Spatial Smoothing ###")
        print("### ResNet18 ###")
        #spatialsmoothingTest(path_resnet18, val_batches, device, resnet18)

        print("### ResNet50 ###")
        #spatialsmoothingTest(path_resnet50, val_batches, device, resnet50)


        print("### Adversarial Training ###")
        print("### Generate Data for Adversarial Training on ResNet18 ###")
        filename_resnet18 = "./data/data_advtrain/datar18.tar"

        gen_adv(train_batches, device, resnet18, path_resnet18, filename_resnet18)

        print("### Generate Data for Adversarial Training on ResNet50 ###")
        filename_resnet50 = "./data/data_advtrain/datar50.tar"
        #gen_adv(train_batches, device, resnet50, path_resnet50, filename_resnet50)

        print("### ResNet18 ###")

        path_resnet18_adv_train = "./model/resnet18_adv_train.pth"

        modelname_resnet18_adv_train = "_resnet18_adv_train_"

        AdversarialTraining(resnet18, train_batches, device, path_resnet18_adv_train, modelname_resnet18_adv_train, filename_resnet18)

        print("### ResNet50 ###")

        path_resnet50_adv_train = "./model/resnet50_adv_train.pth"

        modelname_resnet50_adv_train = "_resnet50_adv_train_"

        #AdversarialTraining(resnet50, train_batches, device, path_resnet50_adv_train, modelname_resnet50_adv_train)


        print("### Testing ###")
        print("### Normal Testing on ResNet18 ###")

        modelname_resnet18_at_test = "_resnet18_at_test_"
        test(resnet18, val_batches, device, path_resnet18_adv_train, modelname_resnet18_at_test)

        print("### Testing with Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_at_aa = "_resnet18_test_at_aa_"
        adversarialattack(path_resnet18_adv_train, val_batches, device, resnet18, modelname_resnet18_test_at_aa)

        print("### Normal Testing on ResNet50 ###")

        modelname_resnet50_at_test = "_resnet50_at_test_"
        test(resnet50, val_batches, device, path_resnet50_adv_train, modelname_resnet50_at_test)

        print("### Testing with Adversarial Attack on ResNet18 ###")
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
        """print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")
        print("### Vanilla DCGAN ###")
        output = torch.load("./outputs/metrics/_Van_GAN_Checkpoint.pth")
        print(output)

        print("### Gradient Penalty ###")
        output = torch.load("./outputs/metrics/_GP_GAN_Checkpoint.pth")
        print(output)

        print("### Weight Clipping (WGAN) ###")
        output = torch.load("./outputs/metrics/_Clipping_GAN_Checkpoint.pth")
        print(output)

        print("### Imbalanced Training (WGAN-GP) ###")
        output = torch.load("./outputs/metrics/_IT_GAN_Checkpoint.pth")
        print(output)

        print("### Layer-Output Normalization (Instance Normalization) ###")
        output = torch.load("./outputs/metrics/_LN_GAN_Checkpoint.pth")
        print(output)

        print("### xAI-LDGAN ###")
        output = torch.load("./outputs/metrics/_LDGAN_Checkpoint.pth")
        print(output)"""

device = get_device()
resnet18 = ResNet18().to(device)
resnet50 = ResNet50().to(device)
project_main(device, resnet18, resnet50)
