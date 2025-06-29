import torch
import torchvision.utils as vutils
import os
from model import Generator


checkpoint_path = r"checkpoints/best.pth"
output_dir = "generated_images"
ngpu = 1
nz = 100
ngf = 64
nc = 3
num_images = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


netG = Generator(ngpu, nz, ngf, nc).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.eval()


fixed_noise = torch.randn(num_images, nz, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()


os.makedirs(output_dir, exist_ok=True)


vutils.save_image(fake_images, os.path.join(output_dir, "image_generator.png"), nrow=8, normalize=True)


