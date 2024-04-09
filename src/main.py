#Import runtime libraries
import os
import sys

# Append needed function/module paths
sys.path.append('./src')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/new_resnet')
sys.path.append('./src/dataloader/dataloadernew')
sys.path.append('./src/metrics/')
sys.path.append('./src/metrics/acc')
sys.path.append('./src/training')
sys.path.append('./src/training/train')
sys.path.append('./src/testing')
sys.path.append('./src/testing/testing')
sys.path.append('./src/testing/attacks')
sys.path.append('./src/defensemethod/')
sys.path.append('./src/defensemethod/adversarialtraining')
sys.path.append('./src/defensemethod/spatialsmoothing')
sys.path.append('./src/utils/')
sys.path.append('./src/utils/utils')
sys.path.append('./src/utils/config')

#custom moduls
from utils import get_device
from new_resnet import ResNet50, ResNet18
from train import train
import dataloadernew
from testing import test
from attacks import adversarialattack
from adversarialtraining import AdversarialTraining
from spatialsmoothing import spatialsmoothingTest, spatialsmoothingTest_withoutattack
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
    print("3. Adversarial Attack on ResNet18 and ResNet50")
    print("4. Spatial Smoothing - ResNet18 and ResNet50")
    print("5. Adversarial Training - ResNet18 and ResNet50")
    print("6. Adversarial Training and Spatial Smoothing - ResNet18 and  ResNet50")
    print("##########")
    print("7. Call Checkpoints (Metrics - Losses and Acc) of all Models")
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
        
        # Defense Method
        print("\n")
        print("### Combine ResNet18 Adversarial Training with Spatial Smoothing ###")
        modelname_resne18_test_at_smoothing = "_resnet18_at_smoothing_"
        spatialsmoothingTest(path_resnet18_adv_train, val_batches, device, resnet18,modelname_resne18_test_at_smoothing)

        
        print("\n")
        print("### Normal Testing on ResNet50 ###")
        modelname_resnet50_at_test = "_resnet50_at_test_"
        test(resnet50, val_batches, device, path_resnet50_adv_train, modelname_resnet50_at_test)

        print("\n")
        print("### Testing with Adversarial Attack on ResNet50 ###")
        modelname_resnet50_test_at_aa = "_resnet50_test_at_aa_"
        adversarialattack(path_resnet50_adv_train, val_batches, device, resnet50, modelname_resnet50_test_at_aa)
        
        print("\n")
        print("### Combine ResNet50 Adversarial Training with Spatial Smoothing ###")
        modelname_resne50_test_at_smoothing = "_resnet50_at_smoothing_"
        spatialsmoothingTest(path_resnet50_adv_train, val_batches, device, resnet50,modelname_resne50_test_at_smoothing)
    

    elif user_input == 1:
        cls()
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        
        print("\n")
        print("### Training ResNet18 ###")
        path_resnet18 = "./model/resnet18.pth"
        modelname_resnet18 = "_resnet18_"
        train(resnet18, train_batches, device, path_resnet18, modelname_resnet18)
        
        print("\n")
        print("### Testing ResNet18 ###")
        modelname_resnet18_test = "_resnet18_test_"
        test(resnet18, val_batches, device, path_resnet18, modelname_resnet18_test)

    elif user_input == 2:
        cls()
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)

        print("\n")
        print("### Training Resnet50 ###")
        path_resnet50 = "./model/resnet50.pth"
        modelname_resnet50 = "_resnet50_"
        train(resnet50, train_batches, device, path_resnet50, modelname_resnet50)

        print("\n")
        print("### Testing ResNet50 ###")
        print("### Normal Testing on ResNet50 ###")
        modelname_resnet50_test = "_resnet50_test_"
        test(resnet50, val_batches, device, path_resnet50, modelname_resnet50_test)

    elif user_input == 3:
        cls()
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        print("\n")
        path_resnet18 = "./model/resnet18.pth"
        path_resnet50 = "./model/resnet50.pth"

        print("### Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_aa = "_resnet18_test_aa_"
        adversarialattack(path_resnet18, val_batches, device, resnet18, modelname_resnet18_test_aa)

        print("\n")
        print("### Adversarial Attack on ResNet50 ###")
        modelname_resnet50_test_aa = "_resnet50_test_aa_"
        adversarialattack(path_resnet50, val_batches, device, resnet50, modelname_resnet50_test_aa)
        
    elif user_input==4:
        cls()
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        path_resnet18 = "./model/resnet18.pth"
        path_resnet50 = "./model/resnet50.pth"
        # Defense Method
        print("\n")
        print("### Spatial Smoothing ###")
        print("### ResNet18 with Adversarial Attack ###")
        modelname_resne18_test_smoothing = "_resnet18_smoothing_"
        #spatialsmoothingTest(path_resnet18, val_batches, device, resnet18 ,modelname_resne18_test_smoothing)
        
        print("### ResNet18 without Adversarial Attack ###")
        modelname_resne18_test_smoothing_withoutattack = "_resnet18_smoothing_woattack"
        spatialsmoothingTest_withoutattack(path_resnet18, val_batches, device, resnet18 ,modelname_resne18_test_smoothing_withoutattack)

        print("\n")
        print("### ResNet50 with Adversarial Attack ###")
        modelname_resne50_test_smoothing = "_resnet50_smoothing_"
        #spatialsmoothingTest(path_resnet50, val_batches, device, resnet50, modelname_resne50_test_smoothing)
        
        print("### ResNet50 without Adversarial Attack ###")
        modelname_resne50_test_smoothing_withoutattack = "_resnet50_smoothing_woattack"
        spatialsmoothingTest_withoutattack(path_resnet50, val_batches, device, resnet50 ,modelname_resne50_test_smoothing_withoutattack)

    elif user_input == 5:
        cls()
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        path_resnet18 = "./model/resnet18.pth"
        path_resnet50 = "./model/resnet50.pth"
        print("\n")
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

        print("\n")
        print("### Testing with Adversarial Attack on ResNet18 ###")
        modelname_resnet18_test_at_aa = "_resnet18_test_at_aa_"
        adversarialattack(path_resnet18_adv_train, val_batches, device, resnet18, modelname_resnet18_test_at_aa)
        
        print("\n")
        print("### Testing with Adversarial Attack on ResNet50 ###")
        modelname_resnet50_test_at_aa = "_resnet50_test_at_aa_"
        adversarialattack(path_resnet50_adv_train, val_batches, device, resnet50, modelname_resnet50_test_at_aa)

    elif user_input == 6:
        cls()
        
        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)
        path_resnet18_adv_train = "./model/resnet18_adv_train.pth"
        path_resnet50_adv_train = "./model/resnet50_adv_train.pth"

        
        print("\n")
        print("### Combine ResNet18 Adversarial Training with Spatial Smoothing ###")
        modelname_resne18_test_at_smoothing = "_resnet18_at_smoothing_"
        #spatialsmoothingTest(path_resnet18_adv_train, val_batches, device, resnet18,modelname_resne18_test_at_smoothing)

        modelname_resne18_test_at_smoothing_woattack= "_resnet18_at_smoothing_woattack"
        spatialsmoothingTest_withoutattack(path_resnet18_adv_train, val_batches, device, resnet18 ,modelname_resne18_test_at_smoothing_woattack)

        
        print("\n")
        print("### Combine ResNet50 Adversarial Training with Spatial Smoothing ###")
        modelname_resne50_test_at_smoothing = "_resnet50_at_smoothing_"
       #spatialsmoothingTest(path_resnet50_adv_train, val_batches, device, resnet50,modelname_resne50_test_at_smoothing)
        
        
        modelname_resne50_test_at_smoothing_woattack= "_resnet50_at_smoothing_woattack"
        spatialsmoothingTest_withoutattack(path_resnet50_adv_train, val_batches, device, resnet50 ,modelname_resne50_test_at_smoothing_woattack)
    

    elif user_input == 7:
        cls()
        print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")
        
        print("Trainingscheckpoints")
        
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
          
        
        print("Testcheckpoints")
        print("### ResNet18 ###")
        modelname_resnet18_test = "_resnet18_test_"

        output = torch.load("./model/metrics/"+modelname_resnet18_test+"test_Checkpoint.pth", map_location='cpu')
        print(output)
        
        print("### ResNet50 ###")
        modelname_resnet50_test = "_resnet50_test_"

        output = torch.load("./model/metrics/"+modelname_resnet50_test+"test_Checkpoint.pth", map_location='cpu')
        print(output)
        
        print("### ResNet18 - AA ###")

        output = torch.load("./model/metrics/_resnet18_test_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)
        
        print("### ResNet50 - AA  ###")
        modelname_resnet50_test_aa = "_resnet50_test_aa_"

        output = torch.load("./model/metrics/_resnet50_test_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)       
        
        
        print("### ResNet18 - Spatial Smoothing###")
        modelname_resne18_test_smoothing = "_resnet18_smoothing_"

        output = torch.load("./model/metrics/"+modelname_resne18_test_smoothing+"_test_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet50 - Spatial Smoothing ###")
        modelname_resne50_test_smoothing = "_resnet50_smoothing_"
        output = torch.load("./model/metrics/"+modelname_resne50_test_smoothing+"_test_Checkpoint.pth", map_location='cpu')
        print(output)  
        
        
        print("### ResNet18 with Adversarial Training ###")
        output = torch.load("./model/metrics/_resnet18_test_at_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)
    
    
        print("### ResNet50 with Adversarial Training ###")
        output = torch.load("./model/metrics/_resnet50_test_at_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)
        
        
        print("### ResNet18 with Adversarial Training + Spatial Smoothing###")
        modelname_resne18_test_at_smoothing = "_resnet18_at_smoothing_"
        output = torch.load("./model/metrics/"+modelname_resne18_test_at_smoothing+"_test_Checkpoint.pth", map_location='cpu')
        print(output)
        
        print("### ResNet50 with Adversarial Training + Spatial Smoothing###")
        modelname_resnet50_test_at_smoothing = "_resnet50_at_smoothing_"
        output = torch.load("./model/metrics/"+modelname_resnet50_test_at_smoothing+"_test_Checkpoint.pth", map_location='cpu')
        print(output)
        

device = get_device()
resnet18 = ResNet18().to(device)
resnet50 = ResNet50().to(device)
project_main(device, resnet18, resnet50)
