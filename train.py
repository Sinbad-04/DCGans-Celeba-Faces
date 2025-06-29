<<<<<<< HEAD

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import Generator, Disciminator
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description="Gans")
    parser.add_argument("--data_path", "-d", type=str, default=r"./gans/celeba/img_align_celeba", help="Path to dataset")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rateD", "-lD", type=float, default=0.0001, help="Learning rate for optimizer D")
    parser.add_argument("--learning_rateG", "-lG", type=float, default=0.0003, help="Learning rate for optimizer G")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard", help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="checkpoints", help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    parser.add_argument("--img-sz", "-i", type=int, default=64, help="Image Size")
    parser.add_argument("--nc", "-nc", type=int, default=3, help="Number class")
    parser.add_argument("--ngf", "-ng", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", "-nd", type=int, default=64, help="Size of feature maps in disciminator")
    parser.add_argument("--beta1", "-beta1", type=float, default=0.5)
    parser.add_argument("--ngpu", "-ngpu", type=int, default=1, help="Number of GPU")


    args = parser.parse_args()
    return args


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(args):
    manualSeed = 999


    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)



    data_root = r"celeba/img_align_celeba"
    workers = 2
    batch_size = 32
    img_size = 64
    nc = 3
    nz = 100 #length of talent vector
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in disciminator
    num_epoch = 100
    lrD = 0.0001
    lrG = 0.0003
    beta1 = 0.5
    ngpu = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.isdir("tensorboard"):
        os.makedirs("tensorboard")

    writer = SummaryWriter("tensorboard")

    dataset = dset.ImageFolder(root=data_root,
                               transform= transforms.Compose([
                                   transforms.Resize((args.img_sz, args.img_sz)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=workers)



    # real_batch = next(iter(dataloader))

    netG = Generator(args.ngpu, nz, args.ngf,args.nc).to(device)


    if (device == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))


    netG.apply(weights_init)

    netD = Disciminator(args.ngpu, args.nc, args.ndf).to(device)


    if (device == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))


    netD.apply(weights_init)

    #Loss and optimizer
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 0.9
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rateD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rateG, betas=(args.beta1, 0.999))

    resume = True
    checkpoint_path = 'checkpoints/last.pth'
    start_epoch = 0
    best_lossG = float('inf')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_lossG = checkpoint.get('lossG', float('inf'))


    #training

    img_list = []
    loss_D = []
    loss_G = []
    iters = len(dataloader)

    for epoch in range(start_epoch, args.num_epochs):
        progressbar = tqdm(enumerate(dataloader, 0), colour="cyan")
        running_lossG = 0.0
        for iter, (images, _) in progressbar:
            # Train Discriminator
            netD.zero_grad()

            real_cpu = images.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            #train fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device) #create random vector z for fake img
            fake_img = netG(noise)
            label.fill_(fake_label)

            #distinguish between real and fake images
            output = netD(fake_img.detach()).view(-1)
            lossD_fake = criterion(output, label)

            lossD_fake.backward()
            D_G_z1 = output.mean().item()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            #update G network
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake_img).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()



            global_step = epoch * iters + iter
            writer.add_scalar("Loss/Discriminator", lossD.item(), global_step)
            writer.add_scalar("Loss/Generator", lossG.item(), global_step)
            writer.add_scalar("D(x)", D_x, global_step)
            writer.add_scalar("D(G(z))_before", D_G_z1, global_step)
            writer.add_scalar("D(G(z))_after", D_G_z2, global_step)

            progressbar.set_description(
                f"Epoch: {epoch + 1}/{args.num_epochs}. Iter: {iter}/{iters}. Loss_D: {lossD:.4f}. Loss_G: {lossG:.4f}. D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
            running_lossG += lossG.item()
            if (epoch + 1) % 3 == 0 and iter == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(fake, padding=2, normalize=True)
                writer.add_image("Generated Images", img_grid, global_step)
        avg_lossG = running_lossG / len(dataloader)
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'lossG': avg_lossG
        }, 'checkpoints/last.pth')



        if avg_lossG < best_lossG:
            best_lossG = avg_lossG
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': best_lossG
            }, 'checkpoints/best.pth')


    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)

=======

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import Generator, Disciminator
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description="Gans")
    parser.add_argument("--data_path", "-d", type=str, default=r"./gans/celeba/img_align_celeba", help="Path to dataset")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rateD", "-lD", type=float, default=0.0001, help="Learning rate for optimizer D")
    parser.add_argument("--learning_rateG", "-lG", type=float, default=0.0003, help="Learning rate for optimizer G")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard", help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="checkpoints", help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    parser.add_argument("--img-sz", "-i", type=int, default=64, help="Image Size")
    parser.add_argument("--nc", "-nc", type=int, default=3, help="Number class")
    parser.add_argument("--ngf", "-ng", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", "-nd", type=int, default=64, help="Size of feature maps in disciminator")
    parser.add_argument("--beta1", "-beta1", type=float, default=0.5)
    parser.add_argument("--ngpu", "-ngpu", type=int, default=1, help="Number of GPU")


    args = parser.parse_args()
    return args


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(args):
    manualSeed = 999


    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)



    data_root = r"celeba/img_align_celeba"
    workers = 2
    batch_size = 32
    img_size = 64
    nc = 3
    nz = 100 #length of talent vector
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in disciminator
    num_epoch = 100
    lrD = 0.0001
    lrG = 0.0003
    beta1 = 0.5
    ngpu = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.isdir("tensorboard"):
        os.makedirs("tensorboard")

    writer = SummaryWriter("tensorboard")

    dataset = dset.ImageFolder(root=data_root,
                               transform= transforms.Compose([
                                   transforms.Resize((args.img_sz, args.img_sz)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=workers)



    # real_batch = next(iter(dataloader))

    netG = Generator(args.ngpu, nz, args.ngf,args.nc).to(device)


    if (device == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))


    netG.apply(weights_init)

    netD = Disciminator(args.ngpu, args.nc, args.ndf).to(device)


    if (device == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))


    netD.apply(weights_init)

    #Loss and optimizer
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 0.9
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rateD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rateG, betas=(args.beta1, 0.999))

    resume = True
    checkpoint_path = 'checkpoints/last.pth'
    start_epoch = 0
    best_lossG = float('inf')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_lossG = checkpoint.get('lossG', float('inf'))


    #training

    img_list = []
    loss_D = []
    loss_G = []
    iters = len(dataloader)

    for epoch in range(start_epoch, args.num_epochs):
        progressbar = tqdm(enumerate(dataloader, 0), colour="cyan")
        running_lossG = 0.0
        for iter, (images, _) in progressbar:
            # Train Discriminator
            netD.zero_grad()

            real_cpu = images.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            #train fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device) #create random vector z for fake img
            fake_img = netG(noise)
            label.fill_(fake_label)

            #distinguish between real and fake images
            output = netD(fake_img.detach()).view(-1)
            lossD_fake = criterion(output, label)

            lossD_fake.backward()
            D_G_z1 = output.mean().item()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            #update G network
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake_img).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()



            global_step = epoch * iters + iter
            writer.add_scalar("Loss/Discriminator", lossD.item(), global_step)
            writer.add_scalar("Loss/Generator", lossG.item(), global_step)
            writer.add_scalar("D(x)", D_x, global_step)
            writer.add_scalar("D(G(z))_before", D_G_z1, global_step)
            writer.add_scalar("D(G(z))_after", D_G_z2, global_step)

            progressbar.set_description(
                f"Epoch: {epoch + 1}/{args.num_epochs}. Iter: {iter}/{iters}. Loss_D: {lossD:.4f}. Loss_G: {lossG:.4f}. D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
            running_lossG += lossG.item()
            if (epoch + 1) % 3 == 0 and iter == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(fake, padding=2, normalize=True)
                writer.add_image("Generated Images", img_grid, global_step)
        avg_lossG = running_lossG / len(dataloader)
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'lossG': avg_lossG
        }, 'checkpoints/last.pth')



        if avg_lossG < best_lossG:
            best_lossG = avg_lossG
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': best_lossG
            }, 'checkpoints/best.pth')


    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)

>>>>>>> 72c29a594bdcce9f24e9fe2f51b8519dafb86200
