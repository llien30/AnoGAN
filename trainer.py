import torch
import torch.nn as nn

def train(G, D, dataloader, num_epochs, num_fakeimg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_le, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20
    mini_batch_size = 64

    fixed_z = torch.randn(num_fakeimg, z_dim, 1, 1).to(device)

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    nun_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    