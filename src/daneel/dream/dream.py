# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# CONSTANTS
WORKERS = 2
BATCH_SIZE = 128
IMAGE_SIZE = 32*32
N_CHANNELS = 1
LATENT_SIZE = 8*8
FEATURE_MAPS_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.002

class ImgDataset(Dataset):
    def __init__(self, data_path,label_path, transform = None):
        self.transform = transform
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.data = self.data.reshape(self.data.shape[0],1,self.data.shape[1],self.data.shape[1])
        self.data = self.data[:,:,:32,:32]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        element = self.data[idx,:,:,:]
        if self.transform:
            element = self.transform()  # Apply transformation
        return torch.tensor(element).float(), self.labels[idx]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( LATENT_SIZE, FEATURE_MAPS_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURE_MAPS_SIZE * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(FEATURE_MAPS_SIZE * 8, FEATURE_MAPS_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAPS_SIZE * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( FEATURE_MAPS_SIZE * 4, FEATURE_MAPS_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAPS_SIZE * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( FEATURE_MAPS_SIZE*2, N_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 32 x 32``
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 32 x 32``
            nn.Conv2d(N_CHANNELS, FEATURE_MAPS_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 16 x 16``
            nn.Conv2d(FEATURE_MAPS_SIZE, FEATURE_MAPS_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAPS_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 8 x 8``
            nn.Conv2d(FEATURE_MAPS_SIZE * 2, FEATURE_MAPS_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAPS_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 4 x 4``
            nn.Conv2d(FEATURE_MAPS_SIZE * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def DreamNewWorlds(param, manualSeed = None):
    # Set Seed
    if manualSeed is None:
        manualSeed = np.random.randint(0,1000)
    print("### Dreaming New Worlds ###")
    print("Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    
    data_path = param.get("data_path")
    label_path = param.get("label_path")
        
    # Instantiate Dataset
    dataset = ImgDataset(data_path,label_path,transform=None)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=WORKERS)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Using device", device)
    
    # Create the generator
    netG = Generator()
    netG.apply(weights_init)
    
    # Create the Discriminator
    netD = Discriminator()
    netD.apply(weights_init)
    
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, LATENT_SIZE, 1, 1)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    if param.get("Training"):
        
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(EPOCHS):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0]
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, LATENT_SIZE, 1, 1)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 5 == 0:
                    print('\r[%2d/%2d][%2d/%2d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, EPOCHS, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2),end="")

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
                
                # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    noise = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1)
    # # Generate fake image batch with G
    fake = netG(noise)
    
    if param.get("Training"):
        torch.save(netD.state_dict(), "TrainedDiscriminator.pth")
        torch.save(netG.state_dict(), "TrainedGenerator.pth")
    else:
        netD.load_state_dict(torch.load("TrainedDiscriminator.pth",weights_only=True))
        netG.load_state_dict(torch.load("TrainedGenerator.pth",weights_only=True))
    
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake[:64], padding=2, normalize=True),(1,2,0)))
    plt.show()
    plt.savefig("DREAMS.png")