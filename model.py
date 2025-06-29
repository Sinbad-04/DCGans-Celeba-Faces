<<<<<<< HEAD
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # [ngf*8, 4, 4]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # [ngf*4, 8, 8]
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # [ngf*2, 16, 16]
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # [ngf, 32, 32]
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # [nc, 64, 64]
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)


class Disciminator(nn.Module):

    def __init__(self, ngpu, nc, ndf):
         super(Disciminator, self).__init__()
         self.ngpu = ngpu
         self.main = nn.Sequential(
             nn.Conv2d(in_channels= nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*2),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*4),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*8),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
             nn.Sigmoid()

     )
    def forward(self, input):
=======
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # [ngf*8, 4, 4]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # [ngf*4, 8, 8]
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # [ngf*2, 16, 16]
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # [ngf, 32, 32]
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # [nc, 64, 64]
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)


class Disciminator(nn.Module):

    def __init__(self, ngpu, nc, ndf):
         super(Disciminator, self).__init__()
         self.ngpu = ngpu
         self.main = nn.Sequential(
             nn.Conv2d(in_channels= nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*2),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*4),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(ndf*8),
             nn.LeakyReLU(0.2, inplace=True),

             nn.Conv2d(in_channels= ndf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
             nn.Sigmoid()

     )
    def forward(self, input):
>>>>>>> 72c29a594bdcce9f24e9fe2f51b8519dafb86200
        return self.main(input)