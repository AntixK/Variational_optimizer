import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self._name = 'cifarG'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        self.main = nn.Sequential(
                nn.Linear(self.dim, 4 * 4 * 4 * self.dim),
                nn.BatchNorm2d(4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.dim, 3, 2, stride=2),
                nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 3, 32, 32)

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self._name = 'cifarE'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        self.main = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
        self.linear = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output.view(-1, self.dim)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self._name = 'cifarD'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        self.main = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                )
        self.linear = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output