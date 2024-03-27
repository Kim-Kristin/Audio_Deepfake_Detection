#https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial
#https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-pytorch/4-speech-model

# Import numpy library for numerical operations
import numpy as np
# Import torch library for building and training neural networks
import torch
# Import nn module from torch for building neural network layers
from torch import nn
# Import torch multiprocessing module for parallel processing
import torch.multiprocessing
# Import datasets and transforms modules from torchvision for loading and transforming image datasets
from torchvision import datasets, transforms
# Import torchvision library for image processing
import torchvision
# Import pyplot module from matplotlib for plotting graphs
import matplotlib.pyplot as plt
# Import tqdm module for displaying progress bars
from tqdm.auto import tqdm
# Import default_timer function from timeit for measuring time taken for model training
from timeit import default_timer as timer
import os

import pandas as pd
import matplotlib.pyplot as plt
import librosa

import seaborn as sns
from tqdm import tqdm

from torch.optim import Adam
#import torchaudio
from scipy.io import wavfile
#from python_speech_features import mfcc, logfbank
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim


import sys
sys.path.append('./src')
sys.path.append('./src/utils/')
sys.path.append('./src')
sys.path.append('./src/metrics/')
sys.path.append('./src/metrics/acc')

sys.path.append('./src/utils/utils')
sys.path.append('./src/utils/')
sys.path.append('./src/utils/config')
sys.path.append('./src')

sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/dataloader')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/spectogram')

sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/new_resnet')
sys.path.append('./src/dataloader')
sys.path.append('./src/dataloader/ResNet_train')

sys.path.append('./src')
sys.path.append('./src/metrics/')
sys.path.append('./src/metrics/acc')

sys.path.append('./src/testing')
sys.path.append('./src/testing/attacks')


sys.path.append('./src')
sys.path.append('./src/defensemethod/')
sys.path.append('./src/defensemethod/adversarialtraining')
sys.path.append('./src/defensemethod/spatialsmoothing')


sys.path.append('./data')
sys.path.append('./data/flac')

from new_resnet import ResNet50
import acc
import attacks
import adversarialtraining
from spatialsmoothing  import median_smoothing, gaussianblur, spatialsmoothingTest

import utils
from config import config
from spectogram import create_spectrogram_images, show_spectrogram, audio

import warnings
warnings.filterwarnings('ignore')

#Variables

NUM_WORKERS = 0 # number of worker used when loading data into dataloader
DATASET_PATH = './data/spec/' # path of our spectrogram dataset
IMAGE_SIZE = (1024, 1024) # image size
CHANNEL_COUNT = 3 # 3 channel as an image has 3 color (R,G,B)
ATTRIBUTION = ["spoof", "bonafide"] # class labels exemple, we'll have 3 class in this exemple
ACCURACY_THRESHOLD = 98 # accuracy at which to stop
batch_size = 16


# Define the data transformation, we will only use it to transform the image as tensor
# adding noise, pitch shifting, time stretching are valid transformations that we you could use
# see https://pytorch.org/audio/stable/transforms.html
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
    # read csv
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

    #print(df_clean.lenght)
    classes = list(np.unique(df_clean.label))
    class_dist = df_clean.groupby(['label'])['lenght'].mean()
    print("Unique Classes: ", classes)
    
    fig, ax = plt.subplots()
    ax.set_title('Original Data: Class Distribution of Audiolenght', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
        shadow=False, startangle=90)
    ax.axis('equal')
    #plt.show()
    plt.savefig("./data/outputs/dataset/orgdata_classdist.png")
    plt.close()
    

    
    path_clean = './data/clean/'
    if os.path.exists(path_clean) is False:
            os.mkdir(path_clean)

    path_clean_s = './data/clean/spoof/'
    if os.path.exists(path_clean_s) is False:
            os.mkdir(path_clean_s)
    path_clean_b = './data/clean/bonafide/'
    if os.path.exists(path_clean_b) is False:
            os.mkdir(path_clean_b)


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

    for c in classes:
        #print(c)
        wav_file = df_clean[df_clean.label == c].iloc[0,0] #test on one sample for each class
        signal, rate = librosa.load(path_audio+wav_file+'.flac', sr=44100)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal
        #fft[c] = calc_fft(signal, rate)

        #bank = logfbank(signal[: rate], rate, nfilt=26, nfft=1103).T
        #fbank[c] = bank

        #mel = mfcc(signal[: rate], rate, numcep=13, nfilt=26, nfft=1103).T
        #mfccs[c] = mel

    plot_signals(signals)
    plt.savefig("./data/outputs/dataset/bon_spoof_ts.png")
    plt.close()
    #plt.show()
                
    input_path_b = path_clean_b
    input_path_s = path_clean_s

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
    if len(os.listdir(output_path_s)) == 0:
        for f in tqdm(df_clean.fName):
                if df_clean.label[f] == "spoof":
                    sound = audio(input_path_s+f+'.flac')
                    sound.write_disk_spectrogram(output_path_s+f+".png", dpi=SPECTROGRAM_DPI)
                else:
                    sound = audio(input_path_b+f+'.flac')
                    sound.write_disk_spectrogram(output_path_b+f+".png", dpi=SPECTROGRAM_DPI)

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

    #Traindataset
    """trainx = []
    trainy = []
    for i, data in enumerate(train_dataset):
        image, label = data
        trainx.append(image.numpy())
        trainy.append(np.array(label))

    # Testdataset
    testx = []
    testy = []
    for i, data in enumerate(val_dataset):
        image, label = data
        testx.append(image.numpy())
        testy.append(np.array(label))"""

    # Load training dataset into batches
    train_batches = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    # Load validation dataset into batches
    val_batches = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size*2,
                                            num_workers=NUM_WORKERS)

    # display 32 (batch_size*2) sample from the first validation batch
    batches_display(val_batches)

    return train_batches, val_batches


# Calculates accuracy between truth labels and predictions.
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc