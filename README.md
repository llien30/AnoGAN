# AnoGAN

## About CONFIG file

.yaml file must be written in the fallowing format.

### for training

```.yaml
# about dataset
train_csv_file: ./csv/train.csv
input_size: 64
batch_size: 16

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

### for anomaly detection

```.yaml
name: mnist78-2020-02-12_test

test_csv_file: ./csv/test.csv
test_save_dir: ./result

input_size: 64
# test parameters
test_batch_size: 5
z_dim: 20 # same as the training z_dim
channel: 1
ngf: 64 # same as the input_size
ndf: 64 # same as the input_size
extralayer: 0

name: first_test
```

## Attention

If you want to change the project at **wandb**, you have to change *"project name"* in **train.py**
