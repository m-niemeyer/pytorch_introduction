
# coding: utf-8

# In[1]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# In[4]:


# Root directory for dataset
dataroot = "./data/celebra"

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
print("Using", torch.cuda.device_count(), "GPUs.")
ngpu = torch.cuda.device_count()

# In[5]:


# We can use an image folder dataset the way we have it setup.
# Create the dataset
print("Loading Dataset...")
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
print("Done.")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# In[6]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        


# In[11]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # In-Channel, Out-Channel, kernel-size, stride, padding
            nn.BatchNorm2d(ngf * 8), # The number of channels
            nn.ReLU(True), # 4x4
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True), # 8x8
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True), # 16x16
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True), # 32x32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # 64x64
        )
    def forward(self, input):
        #print('Generator Inside Input', input.size())
        output = self.main(input)
        #print('Generator Inside Output', output.size())
        return output
        


# In[12]:




# In[13]:

netG = Generator(ngpu)
if (device.type == "cuda") and (ngpu > 1):
    #netG = nn.DataParallel(netG, list(range(ngpu)))
    netG = nn.DataParallel(netG)

netG = netG.to(device)

# In[14]:


netG.apply(weights_init)


# In[16]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)


# In[17]:


netD = Discriminator(ngpu)

if (device.type == "cuda" and (ngpu > 1)):
    netD = nn.DataParallel(netD)

netD = netD.to(device)

# In[19]:


netD.apply(weights_init)


# In[20]:


criterion = nn.BCELoss()


# In[21]:


fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0


# In[23]:


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[24]:


img_list = []
G_losses = []
D_losses = []
iters = 0


# In[32]:


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


# In[34]:


print("Starting Training loop ...")
for epoch in range(num_epochs):
    time_0 = time.time()
    for i, data in enumerate(dataloader, 0):
        # First, Discriminator Update
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        #print('Input Generator', noise.size())
        fake = netG(noise)
        #print('Output Generator', fake.size())
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1) # WHY DETACH
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # Next, Generator Update
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        # Training Stats
        if i % 50 == 0:
            time_1 = time.time()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tTime: %4.4f'
                 % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, time_1-time_0))
            time_0 = time.time()
            
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        #if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            #with torch.no_grad():
                #fake = netG(fixed_noise).detach().cpu()
                #current_imgs = vutils.make_grid(fake, padding=2, normalize=True)
                #file_name = "./output/img_" + str(epoch+1) + "_" + str(iters) + ".png"
                #vutils.save_image(fake, file_name, padding=2, normalize=True)
                #img_list.append(current_imgs) 
        iters +=1

#netG.save(model.state_dict(), './models/netG.pt')
#netD.save(model.state_dict(), './models/netD.pt')

torch.save(netG.module.state_dict(), './models/netG.pt')
torch.save(netD.module.state_dict(), './models/netD.pt')
