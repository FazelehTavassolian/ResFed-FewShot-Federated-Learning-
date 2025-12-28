# ResFed-FewShot-Federated-Learning-
 ‎This is implementation code of ResFed algorithem, a federated meta-learning method specifically tailored for Few-shot ‎‎classification. The approach involves leveraging a pre-trained model and implementing data ‎augmentation within the federated meta-learner, leading to favorable performance outcomes.



# Installation
## Environment
### Anaconda
Install anaconda for your platform

### Cuda Driver
You should be sure that you cuda driver has been installed or updated before.

### Create an Environment Variable and install packages
> conda create -n your-env-name python=3.7

### Activate Environment
> conda activate your-env-name

### Install Packages
> pip install requirements.txt


## Initialize Clients

> python train.py --make_dataset


## Train the Federated Engine
> python train.py
