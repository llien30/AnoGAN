import torch
import torch.nn as nn

def train(G, D, dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9

    g_optimizer = 