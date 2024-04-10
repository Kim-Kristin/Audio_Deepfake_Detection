# Audio_Deepfake_Detection
Repository for Master Thesis about the Topic Abwehr von State-of-the-Art Black-Box Adversarial Attacks auf Audio Deepfake Detektionmodelle mittels Adversarial Training und Spatial Smoothing

This repository contains the code moduls and data for the assignment with the same Topic.

### Abstract

This master thesis investigates the topic of defense against black-box adversarial attacks on audio deepfake detection models. Existing research in this area shows that various detection models exist, with residual neural networks (ResNet) being frequently used in this context as they offer a high potential for success in detecting audio deepfakes. However, due to their architecture, they are susceptible to adversarial attacks that are indistinguishable from the original datasets. A good audio deepfake detection model should not only be robust against synthetic and converted audio data, but also counteract malicious attacks.
The master thesis therefore proposes a solution for this task, which attempts to reduce the vulnerability of ResNet architectures (ResNet-18 and ResNet-50) for the detection of synthetic audio data, against black box attacks by means of adversarial training, spatial smoothing and a combination of both defense strategies. The models are trained and evaluated on the ASVSpoof sub-dataset "Speech Deepfake" from 2021. The results show that these two defense methods increase the robustness of both detection models under the used black-box adversarial attack "Pixle". The best results for both detection models are achieved with the combined defense strategy. An accuracy of 91.89% is achieved for the ResNet-18, while a comparison with the base models without defense strategy achieves an accuracy of 25.16%. An accuracy of 93.72 % is achieved for the ResNet-50, without defense this is 23.04 %.

### Components of the solution approach

#### Baseline Models
- ResNet-18
- ResNet-50

#### Adversarial Attack 
- Pixle

#### Defense Method
- Spatial Smoothing (gaussian blur)
- Adversarial Training

#### Dataset - ASVspoof Challenge 2021
- Audio data (flac): The audio data was downloaded from the offical website https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1 (the solution is only based on a small sample not the whole dataset). The downloaded raw audio data can be found [here](data/flac)
- metadata: https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz 

### Main Module
-  [Main to Start the project](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/blob/main/src/main.py)


### Requirements and Installation
To setup your environment, run :

For Linux or MacOS
```
setup script: `./setup.sh` or `sh setup.sh`

```
or for Windows
```
setup script `.\setup.ps1`
```

#### Troubleshooting: If this is not possible, please execute the following commands step by step in your command line from your development environment.
Then activate the python environment:

For Linux or MacOS

```
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
After your environment it setup, run the main.py to start the experiment.


### Results

#### Training accuracy (%) and loss for the baselinemodels and the models with Adversarial Training

![trainacc](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/blob/main/data/results_readme/trainacc.png)
![trainloss](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/blob/main/data/results_readme/trainloss.png)


#### Test accuracy (%) and loss for all models
 
![testaccloss](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/blob/main/data/results_readme/testaccloss.png)

#### Original vs. manipulated (pixle attack) vs. gaussian bluring

![imagesspectrogram](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/blob/main/data/results_readme/gaussiansmoothing.png)


#### States of training and testing of all models
The trained models with and without adversarial training can be found here.
- [Saved_Models](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/tree/main/model)

The saved model states from the testing can be loaded to check the test results. To do this, execute the number 7 in main.py.
- [Checkpoints](https://github.com/Kim-Kristin/Audio_Deepfake_Detection/tree/main/model/metrics)
