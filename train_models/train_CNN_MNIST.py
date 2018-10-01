import torch
import numpy as np
import torch.nn as nn
import random
import argparse
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from Variational_optimizer.data_utils.logg import get_save_path
from Variational_optimizer.models.CNN_MNIST import ConvNet
from Variational_optimizer.data_utils.create_gif import create_gif
#from Variational_optimizer.varoptim.vadam import VAdam
from Variational_optimizer.varoptim.vadam_1 import VAdam

print(torch.__version__)

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=0, help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, default=1943, help='manual seed')
parser.add_argument('--n_class', type=int, default=10, help='number of classes')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum training iterations')
parser.add_argument('--optim', type=str, default='VAdam', help='Optimizer [Adam, VAdam]')
opt = parser.parse_args()
print(opt)

if opt.optim == 'Adam':
    model_name = "CNN_MNIST_Adam"
elif opt.optim =='VAdam':
    model_name = "CNN_MNIST_VAdam"
else:
    raise ValueError("Unknown optimizer name. Please choose from ['Adam', 'VAdam']")

save_model_path, save_result_path, asset_path, data_path = get_save_path(model_name,
                                                                         overwrite=True)
# Hyper parameters
lr = 0.001
batch_size = opt.batchsize

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=data_path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=data_path,
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
model = ConvNet(opt.n_class).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

if opt.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif opt.optim == 'VAdam':
    optimizer = VAdam(model.parameters(), lr=0.0002, betas=(0.5, 0.999),
                       prior_precision=1.0, init_precision=1.0,
                      train_batch_size=len(train_loader.dataset), num_samples=10)
Loss = []
# Train the model
total_step = len(train_loader)
for epoch in range(opt.max_iter):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        if opt.optim == 'Adam':
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            def closure():
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                return loss

            optimizer.step(closure)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, opt.max_iter, i + 1, total_step, loss.item()))
        Loss.append(loss.item())

# do checkpointing
model_filename = save_model_path+"/"+model_name+"_"+str(epoch)
torch.save(model.state_dict(), model_filename)

print(Loss)
np.savetxt(save_model_path+"/Loss.csv", np.asarray(Loss), delimiter=",")

plt.figure()
plt.plot(np.arange(0, len(Loss)), Loss, label= 'CNN_MNIST Loss')
plt.ylabel('Error', fontsize=17)
plt.xlabel('Iteration', fontsize=17)
plt.legend(fontsize=17)
plt.title(model_name+" Error Variation", fontsize=17)
plt.tight_layout()
plt.savefig(asset_path + model_name+'_error_results.png', dpi=400, bbox_inches='tight')
plt.show()
print("Task Completed!")

