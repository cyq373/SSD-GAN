import torch
import torch.optim as optim
import torch_mimicry as mmc
import models.ssd_sngan_32 as ssd_sngan

# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
dataset = mmc.datasets.load_dataset(root='./datasets/', name='cifar100')
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4)

# Define models and optimizers
netG = ssd_sngan.SSD_SNGANGenerator32().to(device)
netD = ssd_sngan.SSD_SNGANDiscriminator32().to(device)
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

# Start training
trainer = mmc.training.Trainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=100000,
    lr_decay='linear',
    dataloader=dataloader,
    log_dir='./log/cifar100',
    device=device)
trainer.train()