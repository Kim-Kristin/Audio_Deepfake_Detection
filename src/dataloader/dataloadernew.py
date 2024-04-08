
#Dataset 
#https://zenodo.org/records/4835108 -> Audio Data .flac
#https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz -> metadata

#Tutorials etc.
#https://github.com/piotrkawa/audio-deepfake-adversarial-attacks/blob/main/src/datasets/deepfake_asvspoof_dataset.py
#https://www.youtube.com/watch?v=gfhx4dr6gJQ&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P&index=9
#https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial
#https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-pytorch/4-speech-model
#https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html#define-the-network
#https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial/notebook
#https://www.youtube.com/watch?v=gfhx4dr6gJQ&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P&index=9



# Import packages and custom functions
import numpy as np
import torch
import torch.multiprocessing
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


import sys
sys.path.append('./src')
sys.path.append('./src/utils/')
sys.path.append('./src/metrics/')
sys.path.append('./src/metrics/acc')
sys.path.append('./src/utils/utils')
sys.path.append('./src/utils/')
sys.path.append('./src/utils/config')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/dataloader')
sys.path.append('./src/dataloader/spectogram')
sys.path.append('./src/dataloader/new_resnet')
sys.path.append('./src/dataloader/ResNet_train')
sys.path.append('./src/defensemethod/')
sys.path.append('./src/defensemethod/spatialsmoothing')
sys.path.append('./data')
sys.path.append('./data/flac')

from new_resnet import ResNet50
import acc

import utils
from config import config
from spectogram import audio

import warnings
warnings.filterwarnings('ignore')

#Variables
NUM_WORKERS = 0 
DATASET_PATH = './data/spec/' 
batch_size = 16

# Define the data transformation -> https://pytorch.org/audio/stable/transforms.html
transform=transforms.Compose([transforms.Resize((32,32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y.flatten()).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

"""
Display a spectrogram image
@param img: Spectrogram of sound
@param one_channel: Whenever image is grey or has color (RGB)
"""
def image_display_spectrogram(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
Display all the spectrogram of sounds within a batch
@param batches: Batch of data from a dataloader
"""
def batches_display(batches):
    dataiter = iter(batches)
    images, _ = next(dataiter)
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    image_display_spectrogram(img_grid, one_channel=False)
    
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize= (20,10))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        axes[x].set_title(list(signals.keys())[i])
        axes[x].plot (list(signals. values ())[i])
        axes[x].get_xaxis().set_visible (False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def dataset(device):
  
    # readinag given csv file 
    # and creating dataframe 
    org_txt = pd.read_csv("./data/DF/CM/trial_metadata.txt", delimiter = "\t") 
    print("Length Orginal txt-File: ", len(org_txt))
    files = os.listdir('./data/flac')
    # open file in write mode
    with open(r'./data/DF/file.txt', 'w') as fp:
        print("Extract Name of Audiofile")
        for item in files:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    
    #load metadata and extract important information
    df_org = pd.read_csv("./data/DF/CM/trial_metadata2.csv", header= None, sep=";")
    df_clean = pd.DataFrame()
    df_clean['fName'] = df_org.iloc[:,1].values
    df_clean['label'] = df_org.iloc[:,5].values
    df_clean.set_index(df_clean.fName.values, inplace=True)

    # read audio files
    path_audio = './data/flac/'
    for f in tqdm(df_clean.index):
        filename = path_audio+f+".flac"
        duration = librosa.get_duration(path=filename)
        rate, signal = librosa.load(filename)
        df_clean.loc[f,'lenght'] = duration
    
    #extract unique classes
    classes = list(np.unique(df_clean.label))
    class_dist = df_clean.groupby(['label'])['lenght'].mean()
    print("Unique Classes: ", classes)
    
    # plot class distribution (audiolength)
    fig, ax = plt.subplots()
    ax.set_title('Original Data: Class Distribution of Audiolenght', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
        shadow=False, startangle=90)
    ax.axis('equal')
    plt.savefig("./data/outputs/dataset/orgdata_classdist.png")
    plt.close()
    
    signals_org = {}
    path_audio_org= "./data/flac/"

    #routine for plotting audiowave
    for c in classes:
        #print(c)
        wav_file = df_clean[df_clean.label == c].iloc[0,0] #test on one sample for each class
        signal, rate = librosa.load(path_audio_org+wav_file+'.flac', sr=44100)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals_org[c] = signal

    plot_signals(signals_org)
    plt.savefig("./data/outputs/dataset/bon_spoof_org_ts.png")
    plt.close()

    #clean audio directory
    path_clean = './data/clean/'
    if os.path.exists(path_clean) is False:
            os.mkdir(path_clean)

    path_clean_s = './data/clean/spoof/'
    if os.path.exists(path_clean_s) is False:
            os.mkdir(path_clean_s)
    path_clean_b = './data/clean/bonafide/'
    if os.path.exists(path_clean_b) is False:
            os.mkdir(path_clean_b)

    #sort audiofiles in clean directory
    if len(os.listdir(path_clean_s)) == 0:
        for f in tqdm(df_clean.fName):
            if df_clean.label[f] == "spoof":
                signal, rate = librosa.load(path_audio+f+'.flac', sr=16000)
                mask = envelope(signal, rate, 0.0005)
                wavfile.write(filename=path_clean_s+f+'.flac', rate=rate, data=signal[mask])

            else:
                signal, rate = librosa.load(path_audio+f+'.flac', sr=16000)
                mask = envelope(signal, rate, 0.0005)
                wavfile.write(filename=path_clean_b+f+'.flac', rate=rate, data=signal[mask])
        
    signals = {}
    #routine for plotting audiowave
    for c in classes:
        wav_file = df_clean[df_clean.label == c].iloc[0,0] #test on one sample for each class
        signal, rate = librosa.load(path_audio+wav_file+'.flac', sr=44100)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal


    plot_signals(signals)
    plt.savefig("./data/outputs/dataset/bon_spoof_ts.png")
    plt.close()
                
    input_path_b = path_clean_b
    input_path_s = path_clean_s

    # spectrogram directory
    path_images = './data/spec/'
    if os.path.exists(path_images) is False:
            os.mkdir(path_images)

    path_images_s = './data/spec/spoof/'
    if os.path.exists(path_images_s) is False:
            os.mkdir(path_images_s)
    path_images_b = './data/spec/bonafide/'
    if os.path.exists(path_images_b) is False:
            os.mkdir(path_images_b)

    output_path_s = path_images_s
    output_path_b = path_images_b
    SPECTROGRAM_DPI = 90 # image quality of spectrograms
    
    #make spectrograms and sort sprectrogram in directory
    if len(os.listdir(output_path_s)) == 0:
        for f in tqdm(df_clean.fName):
                if df_clean.label[f] == "spoof":
                    sound = audio(input_path_s+f+'.flac')
                    sound.write_disk_spectrogram(output_path_s+f+".jpeg", dpi=SPECTROGRAM_DPI)

                else:
                    sound = audio(input_path_b+f+'.flac')
                    sound.write_disk_spectrogram(output_path_b+f+".jpeg", dpi=SPECTROGRAM_DPI)
    
    print(f'Length of spoof dataset: {len(os.listdir(output_path_s))}')
    print(f'Length of bonafide dataset: {len(os.listdir(output_path_b))}')

    # Load the dataset
    print(f"Loading images from dataset at {DATASET_PATH}")
    dataset = torchvision.datasets.ImageFolder('./data/spec/', transform=transform)

    # train / test split
    val_ratio = 0.2
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"{train_size} images for training, {val_size} images for validation")

    # Load training dataset into batches
    train_batches = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            drop_last= True,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS)
    # Load validation dataset into batches
    val_batches = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size*2,
                                            shuffle=True,
                                            drop_last= False,
                                            num_workers=NUM_WORKERS)

    # display 32 (batch_size*2) sample from the first validation batch
    batches_display(val_batches)

    return train_batches, val_batches