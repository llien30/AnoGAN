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

class Discriminator(nn.Module):
    '''
    DCGAN Discriminator network
    '''
    def __init__(self, input_size, z_dim, channel, ndf, n_extra_layers=0, add_final_conv=True):
        super(Discriminator, self).__init__()

        assert input_size%16 == 0, 'input_size has to be a multiple of 16'

        main = nn.Sequential()
        main.add_module('first_conv-{}-{}'.format(channel, ndf),
                        nn.Conv2d(channel, ndf, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('first_BatchNorm-{}'.format(ndf),
                        nn.BatchNorm2d(ndf))
        main.add_module('first_LeakyReLU-{}'.format(ndf),
                        nn.LeakyReLU(inplace=True))
        csize, cndf = input_size//2, ndf


        for _ in range(n_extra_layers):
            main.add_module('extra_conv-{}-{}'.format(cndf, cndf),
                            nn.Conv2d(cndf, cndf, kernel_size=3, stride=1, padding=1, bias=False))
            main.add_module('extra_BatchNorm-{}'.format(cndf),
                            nn.BatchNorm2d(ndf))
            main.add_module('extra_LeakyReLU-{}'.format(cndf),
                            nn.LeakyReLU(inplace=True))
        
        while csize > 4:
            in_feat = cndf
            out_feat = cndf*2
            main.add_module('pyramid_conv-{}-{}'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('pyramid_BatchNorm-{}'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_LeakyReLU-{}'.format(out_feat),
                            nn.LeakyReLU(inplace=True))
            cndf *= 2
            csize //= 2

        if add_final_conv:
            main.add_module('last_conv-{}-{}'.format(cndf, 1),
                            nn.Conv2d(cndf, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)

        return output

class NetD(nn.Module):
    '''
    Discriminator
    '''
    def __init__(self, CONFIG):
        super(NetD, self).__init__()

        model = Discriminator(CONFIG.input_size, CONFIG.z_dim, CONFIG.channel, CONFIG.ndf, CONFIG.extralayer)
        layers = list(model.main.children())

        #to output feature, separate the network
        self.feature = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequentisl(layers[-1])
        #add the normalize layer
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        feature = self.feature(x)
        classifier = self.classifier(feature)
        classifier = classifier.view(-1,1).squeeze(1)

        return classifier, feature


class NetG(nn.Module):
    '''
    Generator
    '''
    def __init__(self, CONFIG):
        super(NetG, self).__init__()
        self.generator = Generator(CONFIG.input_size, CONFIG.z_dim, CONFIG.channel, CONFIG.ngf, CONFIG.extralayer)

    def forward(self, z):
        gan_img = self.generator(z)
        return gan_img



