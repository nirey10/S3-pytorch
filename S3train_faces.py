##!/usr/bin/python3

import argparse
import itertools
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from torchvision.utils import save_image
from S3models import Encoder, Decoder, Identity_Generator, Perceptual
from S3models import Discriminator
from S3utils import ReplayBuffer
from S3utils import LambdaLR
from S3utils import Logger
from S3utils import weights_init_normal
from datasets import ImageDataset, CelebADatasets, CelebBDatasets
import os
import losses as los
from S3utils import get_features


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/celeba/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

RES_DIR = 'output'
OUTPUT_DIR = "faces_test6"

# Create output dirs if they don't exist
if not os.path.exists(RES_DIR + '/' + OUTPUT_DIR):
    os.makedirs(RES_DIR + '/' + OUTPUT_DIR)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

encoder = Encoder(opt.input_nc)
decoder = Decoder()
identity_generator = Identity_Generator(encoder, decoder)
discriminator = Discriminator(opt.input_nc)
perceptual = Perceptual(encoder, decoder, identity_generator)

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    identity_generator.cuda()
    discriminator.cuda()
    perceptual.cuda()

# Initializing weights
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
identity_generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
perceptual.apply(weights_init_normal)

# Lossess
criterion_discriminator = torch.nn.MSELoss()
criterion_adv = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
#optimizer_G = torch.optim.Adam(itertools.chain(identity_generator.parameters()),lr=opt.lr, betas=(0.5, 0.999))
optimizer_Perceptual = torch.optim.Adam(perceptual.parameters(),lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_Perceptual, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
target_real = target_real[:, None]
target_fake = target_fake[:, None]

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

print(opt.dataroot)
# Dataset loader
transforms_ = [ #transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ]

#dataset = datasets.ImageFolder(opt.dataroot, transform=transform)
dataloaderA = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
dataloaderB = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
# dataloaderA = DataLoader(dataset,
#                         batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
# dataloaderB = DataLoader(dataset,
#                         batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    i=-1
    for batchA, batchB in zip(dataloaderA,dataloaderB):
        i+=1
        # Set model input
        real_A = Variable(input_A.copy_(batchA['A']))
        real_B = Variable(input_B.copy_(batchB['B']))

        #real_batch = Variable(input_A.copy_(batch['A']))
        #real_A = torch.unsqueeze(real_batch[0], 0)
        #real_B = torch.unsqueeze(real_batch[1], 0)

        optimizer_Perceptual.zero_grad()

        mixed_image, reconstructionA, reconstructionB = perceptual(real_A, real_B)

        # Reconstruction loss
        loss_reconstruction_A = criterion_identity(reconstructionA, real_A)*30.0
        loss_reconstruction_B = criterion_identity(reconstructionB, real_B)*30.0

        # adv loss
        pred_fake = discriminator(mixed_image)
        loss_adv = criterion_adv(pred_fake, target_real)

        #Total Variational loss
        TV_loss = los.total_variation_loss(mixed_image)

        # Perceptual loss
        cuda_mixed_image = mixed_image.clone().requires_grad_(True).to(device)
        cuda_real_A = real_A.clone().requires_grad_(True).to(device)
        cuda_real_B = real_B.clone().requires_grad_(True).to(device)
        style_features = get_features(cuda_real_A, vgg)
        content_features = get_features(cuda_real_B, vgg)
        target_features = get_features(cuda_mixed_image, vgg)
        content_loss = los.compute_content_loss(target_features['conv4_2'], content_features['conv4_2']) * 0.1
        style_loss = los.compute_style_loss(style_features, target_features) * 0.05

        total_loss = loss_reconstruction_A + loss_reconstruction_B + loss_adv + content_loss + style_loss + TV_loss
        total_loss.backward()
        optimizer_Perceptual.step()

        # Update discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(real_A)
        loss_D_real = criterion_discriminator(pred_real, target_real)
        pred_fake = discriminator(mixed_image.detach())
        loss_D_fake = criterion_discriminator(pred_fake, target_fake)
        total_discriminator_loss = (loss_D_real + loss_D_fake)
        total_discriminator_loss.backward()
        optimizer_D.step()

        print('>%d, %s, weighted[%.3f] reconstruction[%.3f,%.3f]  discriminator[%.3f] perceptual[%.3f,%.3f]' %
              (i + 1, OUTPUT_DIR, total_loss.item(), loss_reconstruction_A.item(), loss_reconstruction_B.item(),
               loss_adv.item(), content_loss.item(), style_loss.item()))

        if  i % 200 == 0:
            real_A = 0.5 * (real_A.data + 1.0)
            real_B = 0.5 * (real_B.data + 1.0)
            mixed_image = 0.5 * (mixed_image.data + 1.0)
            reconstructionA = 0.5 * (reconstructionA.data + 1.0)
            reconstructionB = 0.5 * (reconstructionB.data + 1.0)
            save_image(real_A, 'output/%s/%04d_%d_A.png' % (OUTPUT_DIR, epoch, i))
            save_image(real_B, 'output/%s/%04d_%d_B.png' % (OUTPUT_DIR, epoch, i))
            save_image(reconstructionA, 'output/%s/%04d_%d_reconA.png' % (OUTPUT_DIR, epoch, i))
            save_image(reconstructionB, 'output/%s/%04d_%d_reconB.png' % (OUTPUT_DIR, epoch, i))
            save_image(mixed_image, 'output/%s/%04d_%d_Mixed.png' % (OUTPUT_DIR, epoch, i))

    print("### epoch: " + str(epoch) + " ###")
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(perceptual.state_dict(), RES_DIR + '/' + OUTPUT_DIR + '.pth')

