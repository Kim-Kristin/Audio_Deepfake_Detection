#https://ai.hdm-stuttgart.de/news/2023/ki-im-audiobereich-grundlagen-signalverarbeitung-ml/

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import matplotlib.pyplot as plt


SPECTROGRAM_DPI = 90 # image quality of spectrograms
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOPE_LENGHT = 1024


class audio():
    def __init__(self, filepath_, hop_lenght = DEFAULT_HOPE_LENGHT, samples_rate = DEFAULT_SAMPLE_RATE):
        self.hop_lenght = hop_lenght
        self.samples_rate = samples_rate
        self.waveform, self.sample_rate = torchaudio.load(filepath_)

    def plot_spectrogram(self) -> None:
        waveform = self.waveform.numpy()
        _, axes = plt.subplots(1, 1)
        axes.specgram(waveform[0], Fs=self.sample_rate)
        plt.axis('off')
        plt.show(block=False)
        
    def plot_spectrogram_withaxis(self) -> None:
        waveform = self.waveform.numpy()
        _, axes = plt.subplots(1, 1)
        axes.specgram(waveform[0], Fs=self.sample_rate)
        plt.axis()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig("./data/outputs/Images/example.png", dpi=SPECTROGRAM_DPI, bbox_inches='tight')

    def write_disk_spectrogram(self, path, dpi=SPECTROGRAM_DPI) -> None:
        self.plot_spectrogram()
        #self.plot_spectrogram_withaxis()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close('all')