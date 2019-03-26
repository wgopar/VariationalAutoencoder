import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.learn = 1e-6
        self.mean = 0
        self.logvar = 0
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.fc_mean = nn.Linear(3, 2, bias=True)
        self.fc_stddev = nn.Linear(3, 2, bias=True)
        self.fc3 = nn.Linear(2, 3, bias=True)
        self.fc4 = nn.Linear(3, 4, bias=True)

        self.losses = []
        self.logvars = []
        self.means = []

    def encode(self, x):
        z = torch.tanh(self.fc1(x))
        mean = self.fc_mean(z)
        var = self.fc_stddev(z)
        return mean, var

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        z = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(z))

    def forward(self, x):
        mean, logvar = self.encode(x.view(-1, 4))
        z = self.reparameterize(mean, logvar)
        out = torch.sigmoid(self.decode(z))
        return out, mean, logvar

    def train_model(self, data):
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=self.learn_rate)

        output, self.mean, self.logvar = self.forward(data)
        loss = loss_function(output, data, self.mean, self.logvar)

        self.losses.append(loss.data)
        self.logvars.append(self.logvar.detach().numpy().flatten())
        self.means.append(self.mean.detach().numpy().flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def loss_function(output, x, mean, logvar):
    KLD = 0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar))
    BCE = F.binary_cross_entropy(output, x)
    return KLD + BCE