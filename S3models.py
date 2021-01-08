import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # model += [  nn.Conv2d(512, 512, 4, padding=1),
        #             nn.InstanceNorm2d(512),
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
class Critic(nn.Module):
    def __init__(self, input_nc):
        super(Critic, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # model += [  nn.Conv2d(512, 512, 4, padding=1),
        #             nn.InstanceNorm2d(512),
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1), nn.Flatten(), nn.Linear(196, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        out = x.view(x.size()[0], -1)
        return out

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x

class Discriminator3(nn.Module):
    """ defines a PatchGAN discriminator, adopt from CycleGAN """

    def __init__(self):
        super(Discriminator3, self).__init__()
        use_bias = True

        input_nc = 3
        ndf = 64
        n_layers = 2

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels=input_nc,
                              out_channels=ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(negative_slope=0.2,
                                 inplace=True)]
        # 3       -> 64
        # 512x512 -> 256x256

        nf_mult = 1
        for n in range(1, n_layers):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(in_channels=ndf * nf_mult_prev,
                                   out_channels=ndf * nf_mult,
                                   kernel_size=kw,
                                   stride=2,
                                   padding=padw,
                                   bias=True),
                         nn.InstanceNorm2d(num_features=ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]
            # 64      -> 64 * 2
            # 256x256 -> 128x128
            # 64 * 2  -> 64 * 4
            # 128x128 -> 64x64

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(in_channels=ndf * nf_mult_prev,
                               out_channels=ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=True),
                     nn.InstanceNorm2d(num_features=ndf * nf_mult),
                     nn.LeakyReLU(negative_slope=0.2,
                                  inplace=True)]
        # 64 * 4 -> 64 * 8
        # 64x64  -> 63x63

        """ output 1 channel prediction map """
        sequence += [nn.Conv2d(in_channels=ndf * nf_mult,
                               out_channels=1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]
        # 64 * 8 -> 1
        # 63x63  -> 62x62

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """ standard forward """
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(Encoder, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(128, 256, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(256, 512, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(512, 512, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(1024),
                  nn.ReLU(inplace=True)]

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(1024)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Encoder_noResBlocks(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(Encoder_noResBlocks, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(128, 256, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(256, 512, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(512, 512, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.ReLU(inplace=True)]
        model += [nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(1024),
                  nn.ReLU(inplace=True)]

        # Residual blocks
        #for _ in range(n_residual_blocks):
        #    model += [ResidualBlock(1024)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_nc=1024, output_nc=3):
        super(Decoder, self).__init__()

        model = [nn.ConvTranspose2d(input_nc, 512, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(512),
                  nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.InstanceNorm2d(3),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Identity_Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Identity_Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, A, B):
        latentA = self.encoder(A)
        latentB = self.encoder(B)

        reconstructedA = self.decoder(latentA)
        reconstructedB = self.decoder(latentB)
        return reconstructedA, reconstructedB

class Perceptual(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(Perceptual, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, A, B):

        reconstructedA, reconstructedB = self.generator(A, B)

        latentA = self.encoder(A)
        latentB = self.encoder(B)

        latentA.detach()
        latentB.detach()

        style = latentA[:, 0:512, : , :]
        content = latentB[:, 512:1024, :, :]

        mixed_latent = torch.cat([style, content], dim=1)
        mixed_image = self.decoder(mixed_latent)

        return mixed_image, reconstructedA, reconstructedB


class Perceptual_smaller(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(Perceptual_smaller, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, A, B):

        reconstructedA, reconstructedB = self.generator(A, B)

        latentA = self.encoder(A)
        latentB = self.encoder(B)

        latentA.detach()
        latentB.detach()

        style = latentA[:, 0:128, : , :]
        content = latentB[:, 128:256, :, :]

        mixed_latent = torch.cat([style, content], dim=1)
        mixed_image = self.decoder(mixed_latent)

        return mixed_image, reconstructedA, reconstructedB

class PerceptualRecon(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(PerceptualRecon, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, A, B):

        reconstructedA, reconstructedB = self.generator(A, B)

        latentA = self.encoder(reconstructedA)
        latentB = self.encoder(reconstructedB)

        style = latentA[:, 0:512, : , :]
        content = latentB[:, 512:1024, :, :]

        mixed_latent = torch.cat([style, content], dim=1)
        mixed_image = self.decoder(mixed_latent)

        return mixed_image, reconstructedA, reconstructedB


class Encoder_cycle(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(Encoder_cycle, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 32, 7),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(5):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Encoder_cycle_noRes(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(Encoder_cycle_noRes, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 32, 7),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(5):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        #for _ in range(n_residual_blocks):
        #    model += [ResidualBlock(in_features)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_cycle(nn.Module):
    def __init__(self, input_nc=1024, output_nc=3):
        super(Decoder_cycle, self).__init__()

        in_features = input_nc
        model = []

        out_features = in_features // 2
        for _ in range(5):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(32, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder_cycle_noRes_higherLat(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(Encoder_cycle_noRes_higherLat, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 16, 7),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 16
        out_features = in_features*2
        for _ in range(4):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        #for _ in range(n_residual_blocks):
        #    model += [ResidualBlock(in_features)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_cycle_higherLat(nn.Module):
    def __init__(self, input_nc=256, output_nc=3):
        super(Decoder_cycle_higherLat, self).__init__()

        in_features = input_nc
        model = []

        out_features = in_features // 2
        for _ in range(4):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(16, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)