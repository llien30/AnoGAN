# AnoGAN

## Requirement
Python : 3.x\
PyTorch >= 1.0

You can download the package of Python from `pip install -r requirements.txt`:+1:

## About CONFIG file
.yaml file must be written in the fallowing format.
```
input_size: 64
z_dim: 20
channel: 1
ngf: 64 
ndf: 64 
extralayer: 0

num_epochs: 500
num_fakeimage: 5

save_dir: ./weights

name: first
```

## Attention
If you want to change the project on *wandb*, you have to change project name in *train.py*