# a pytorch model of autoencoder
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim1=256, encoding_dim2=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim1),
            nn.LeakyReLU(),
            nn.Linear(encoding_dim1, encoding_dim2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim2, input_dim)
            # nn.LeakyReLU(),
            # nn.Linear(encoding_dim1, input_dim)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
