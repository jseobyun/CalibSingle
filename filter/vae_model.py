import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class OutlierVAE(nn.Module):
    # https://arxiv.org/pdf/1312.6114.pdf
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        in_channels = 1
        self.z_dim = 4

        # ------------------------------- #
        # 15x15 crop
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            View((-1, 64 * 3 * 3)),
            nn.Linear(64 * 3 * 3, self.z_dim * 2)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64 * 3 * 3),
            View((-1, 64, 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels=in_channels, kernel_size=3, stride=2, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = h[:, :self.z_dim]
        logvar = h[:, self.z_dim:]

        return mu, logvar

    def decode(self, z):
        recons = self.decoder(z)
        return recons

    def reparameterize(self, mu, logvar):
        """
        :mu: [B x D]
        :logvar: [B x D]
        :return: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.empty(std.size()).normal_()

        if mu.is_cuda and not eps.is_cuda:
            eps = eps.to(self.device)
        return mu + eps * std

    def forward(self, x, is_training=False):
        mu, logvar = self.encode(x)
        if is_training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        x_recon = self.decode(z)
        return x_recon, mu, logvar