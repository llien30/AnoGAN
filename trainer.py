import torch
import torch.nn as nn
from torchvision.utils import save_image

import time
from PIL import save_image

import wandb

def train(G, D, dataloader, num_epochs, num_fakeimg, no_wandb):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_le, [beta1, beta2])

    #Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    #the dimention of z
    z_dim = 20 

    #the default mini batch size
    mini_batch_size = 64

    fixed_z = torch.randn(num_fakeimg, z_dim, 1, 1).to(device)

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    nun_train_imgs = len(dataloader.dataset)

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('----------------------(train)----------------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('---------------------------------------------------')

        r_batch_size = 0
        for sample in dataloader:
            r_batch_size = r_batch_size + 1
            '''
            learning Discriminator
            '''
            imges = sample['img']
            if imges.size()[0] == 0:
                continue
            
            imges = imges.to(device)
            mini_batch_size = imges.size()[0]

            label_real = torch.full((mini_batch_size, ),1).to(device)
            label_fake = torch.full((mini_batch_size, ),0).to(device)

            d_out_real, _ = D(imges)

            input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)

            fake_imges = G(input_z)
            d_out_fake, _ = D(fake_imges)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)

            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            '''
            learning Generator
            '''
            input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)

            fake_imges = G(input_z)
            d_out_fake = D(fake_imges)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('---------------------------------------------------')
        print('Epoch {} | | Epoch_D_Loss :{:.4f} || Epoch_G_Loss :{:.4f}'.format(
            epoch, epoch_d_loss/r_batch_size, epoch_g_loss/r_batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        fake_imges = G(fixed_z)
        save_image(fake_imges, 'fake_imges.png')
        
        if not no_wandb:
            wandb.log({
                'train_time': t_epoch_finish - t_epoch_start,
                'd_loss': epoch_d_loss/r_batch_size,
                'g_loss': epoch_g_loss/r_batch_size
            }, step=epoch)

            img = Image.open('fake_images.png')
            wandb.log({
                "image": [wandb.Image(img)]
            }, step=epoch)

            t_epoch_start = time.time()

        
    return G, D
            

