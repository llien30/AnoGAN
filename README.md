# Reimplementation of AnoGAN
This is a reimplementation of AnoGAN that works with MNIST dataset by using pytorch.

AnoGAN : https://arxiv.org/abs/1703.05921


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

## reference
・[Pytorchによる発展ディープラーニング](https://www.amazon.co.jp/%E3%81%A4%E3%81%8F%E3%82%8A%E3%81%AA%E3%81%8C%E3%82%89%E5%AD%A6%E3%81%B6%EF%BC%81PyTorch%E3%81%AB%E3%82%88%E3%82%8B%E7%99%BA%E5%B1%95%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0-%E5%B0%8F%E5%B7%9D-%E9%9B%84%E5%A4%AA%E9%83%8E-ebook/dp/B07VPDVNKW)
(本ではjupyter notebookでコードを紹介していましたが、pythonファイルで再現実装．)
