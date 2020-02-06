import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    DCGAN Generator network
    '''
    def __init__(self, input_size, z_dim, channel, ngf, n_extra_layers=0):
        super(Generator, self).__init__()
        assert input_size%16 == 0, 'input_size has to be a multiple of 16.'

        main = nn.Sequential()
        cngf, tisize = ngf/2, 4
        while tisize != input_size/2:
            cngf *= 2
            tisize *= 2

        main.add_module('initial_convt-{}-{}'.format(z_dim, cngf),
                        nn.ConvTranspose2d(z_dim, cngf, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('initial_BatchNorm-{}-{}'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_ReLU-{}'.format(cngf),
                        nn.ReLU(inplace=True))
        
        csize = 4
        while csize < input_size//2:
            main.add_module('pyramid_convt-{}-{}'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, kernal_size=4, stride=2, padding=1 bias=False))
            main.add_module('pyramid_BatchNorm-{}'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_ReLU-{}'.format(cngf//2),
                            nn.ReLU(inplace=True))
            csize *= 2
            cngf //= 2

        for _ in range(n_extra_layers):
            main.add_module('extra_convt-{}-{}'.format(cngf, cngf),
                            nn.ConvTranspose2d(cngf, cngf, kernel_size=3, stride=1, padding=1, bias=False))
            main.add_module('extra_BatchNorm-{}'.format(cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra_ReLU-{}'.format(cngf),
                            nn.ReLU(inplace=True))
        
        main.add_module('last_convt-{}-{}'.format(cngf, channel),
                        nn.ConvTranspose2d(cngf, channel, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('last_tanh-{}'.format(cngf),
                        nn.Tanh())
        
        self.main = main

    def forward(self, input):
        output = self.main(input)

        return output

