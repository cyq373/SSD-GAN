import torch
import torch.optim as optim
import torch_mimicry as mmc
import models.ssd_sngan_128 as ssd_sngan

# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
dataset = mmc.datasets.load_dataset(root='./datasets/', name='lsun_bedroom_128')
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4)

# Define models and optimizers
netG = ssd_sngan.SSD_SNGANGenerator128().to(device)
netD = ssd_sngan.SSD_SNGANDiscriminator128().to(device)
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

# Start training
trainer = mmc.training.Trainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=2,
    num_steps=100000,
    lr_decay='linear',
    dataloader=dataloader,
    log_dir='./log/lsun_bedroom_128',
    device=device)
trainer.train()
