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
- Spatial Smoothing
- Adversarial Training





# python-template

Precondition:
Windows users can follow the official microsoft tutorial to install python, git and vscode here:

- ​​https://docs.microsoft.com/en-us/windows/python/beginners
- german: https://docs.microsoft.com/de-de/windows/python/beginners

## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) which is a great code editor for many languages like Python and Javascript. The [introduction videos](https://code.visualstudio.com/docs/getstarted/introvideos) explain how to work with VS Code. The [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) provides an introduction about common topics like code editing, linting, debugging and testing in Python. There is also a section about [Python virtual environments](https://code.visualstudio.com/docs/python/environments) which you will need in development. There is also a [Data Science](https://code.visualstudio.com/docs/datascience/overview) section showing how to work with Jupyter Notebooks and common Machine Learning libraries.

The `.vscode` directory contains configurations for useful extensions like [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens0) and [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python). When opening the repository, VS Code will open a prompt to install the recommended extensions.

## Development Setup

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS (see below). This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `requirements.txt`.

### Linux and Mac Users

1. run the setup script: `./setup.sh` or `sh setup.sh`
2. activate the python environment: `source .venv/bin/activate`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

### Windows Users

1. run the setup script `.\setup.ps1`
2. activate the python environment: `.\.venv\Scripts\Activate.ps1`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

Troubleshooting:

- If your system does not allow to run powershell scripts, try to set the execution policy: `Set-ExecutionPolicy RemoteSigned`, see https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/
- If you still cannot run the setup.ps1 script, open it and copy all the commands step by step in your terminal and execute each step
