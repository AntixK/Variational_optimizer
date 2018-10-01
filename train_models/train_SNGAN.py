import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import grad
from time import time
import random
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Variational_optimizer.models.GAN.SNGAN import _netG, _netD
from Variational_optimizer.model_utils.weight_filler import weight_filler
from Variational_optimizer.data_utils.logg import get_save_path
from Variational_optimizer.data_utils.create_gif import create_gif
#from Variational_optimizer.varoptim.vadam import VAdam
from Variational_optimizer.varoptim.vadam_1 import VAdam
import csv

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=0, help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, default=1943, help='manual seed')
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--max_iter', type=int, default=2, help='Maximum training iterations')
parser.add_argument('--optim', type=str, default='VAdam', help='Optimizer [Adam, VAdam]')
opt = parser.parse_args()
print(opt)

print(torch.__version__)

if opt.optim == 'Adam':
    model_name = "SNGAN_Adam"
elif opt.optim =='VAdam':
    model_name = "SNGAN_VAdam"
else:
    raise ValueError("Unknown optimizer name. Please choose from ['Adam', 'VAdam']")

save_model_path, save_result_path, asset_path, data_path = get_save_path(model_name,
                                                                         overwrite=True)
print("All data is saved at :" ,save_model_path)

dataset = datasets.CIFAR10(root=data_path, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(3))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    #torch.cuda.set_device(opt.gpu_ids[2])

cudnn.benchmark = True
n_dis = opt.n_dis
nz = opt.nz

G = _netG(nz, 3, 64)
SND = _netD(3, 64)
print(G)
print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 3, 32, 32)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

fixed_noise = Variable(fixed_noise)
criterion = nn.BCELoss()

if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

if opt.optim == 'Adam':
    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999))
elif opt.optim == 'VAdam':
    optimizerG = VAdam(G.parameters(), lr=0.0002, betas=(0.5, 0.999),
                       prior_precision=1.0, init_precision=1.0,
                       train_batch_size=len(dataloader.dataset), num_samples=10)
    optimizerSND = VAdam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999),
                         prior_precision=1.0, init_precision=1.0,
                         train_batch_size=len(dataloader.dataset), num_samples=10)
print("Number of data points ",len(dataloader.dataset))
print("Number of Generator Parameter", len(list(G.parameters())))
print("Number of Discriminator Parameter", len(list(SND.parameters())))

# Error data
Gen_err = []
Dis_err = []

start_time  = time()

for epoch in range(opt.max_iter):
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
           real_cpu = real_cpu.cuda()
        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        output = SND(inputv)

        errD_real = torch.mean(F.softplus(-output))
        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        D_x = output.data.mean()
        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = SND(fake.detach())
        errD_fake = torch.mean(F.softplus(output))
        #errD_fake = criterion(output, labelv)

        D_G_z1 = output.data.mean()
        #grad_penal = gradient_penalty(inputv.data, SND)
        errD = errD_real + errD_fake #+ grad_penal*10.0

        if opt.optim == 'VAdam':
            def closureSND():
                optimizerSND.zero_grad()
                output = SND(inputv)

                errD_real = torch.mean(F.softplus(-output))

                # train with fake
                fake = G(noisev)
                output = SND(fake.detach())
                errD_fake = torch.mean(F.softplus(output))

                errD = errD_real + errD_fake
                errD.backward(retain_graph=True)
                return errD

            optimizerSND.step(closureSND)
        else:
            errD.backward()
            optimizerSND.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if step % n_dis == 0:
            optimizerG.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = SND(fake)
            errG = torch.mean(F.softplus(-output))
            #errG = criterion(output, labelv)

            D_G_z2 = output.data.mean()

            if opt.optim == 'VAdam':
                def closureG():
                    optimizerG.zero_grad()
                    output = SND(fake)
                    errG = torch.mean(F.softplus(-output))
                    errG.backward(retain_graph=True)
                    return errD

                optimizerG.step(closureG)
            else:

                errG.backward()
                optimizerG.step()

        # Log the error data
        Gen_err.append(errG.data[0])
        Dis_err.append(errD.data[0])

        if i % 20 == 0:
            print('[%3d/%3d][%3d/%3d] Loss_D: %.4f Loss_G: %.4f D(x): %+.4f D(G(z)): %+.4f / %+.4f'
                  % (epoch, 200, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % save_result_path,
                    normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (save_result_path, epoch),
                    normalize=True)

end_time = time()-start_time
# do checkpointing
torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (save_model_path, epoch))
torch.save(SND.state_dict(), '%s/netD_epoch_%d.pth' % (save_model_path, epoch))


np.savetxt(save_model_path+"/Discriminator_error.csv", np.asarray(Dis_err), delimiter=",")
np.savetxt(save_model_path+"/Generator_error.csv", np.asarray(Gen_err), delimiter=",")

# Create Gif
create_gif(save_result_path, model_name, asset_path)

plt.figure()
plt.plot(np.arange(0, len(Gen_err)), Gen_err, label= 'Generator Error')
plt.plot(np.arange(0, len(Dis_err)), Dis_err, label='Discriminator Error')
plt.ylabel('Error', fontsize=17)
plt.xlabel('Iteration', fontsize=17)
plt.legend(fontsize=17)
plt.title(model_name+" Error Variation", fontsize=17)
plt.tight_layout()
plt.savefig(asset_path + model_name+'_error_results.png', dpi=400, bbox_inches='tight')
plt.show()