# AnoGAN

## Requirements
Python : 3.x\
PyTorch >= 1.0

You can download the package of Python from `pip install -r requirements.txt`:+1:

## About CONFIG file
.yaml file must be written in the fallowing format.
#### for training
```
#about dataset
train_csv_file: ./csv/train.csv
input_size: 64

# training hyper parameters
z_dim: 20
channel: 1
ngf: 64 #same as the input_size
ndf: 64 #same as the input_size
extralayer: 0

num_epochs: 500
num_fakeimage: 5

# training output
save_dir: ./weights

# wandb
name: first
```

#### for anomaly detection
```
test_csv_file: ./csv/test.csv
test_save_dir: ./result

# test parameters
test_batch_size: 5
z_dim: 20

name: first_test
```

## :exclamation:Attention
If you want to change the project on **wandb**, you have to change *"project name"* in **train.py**