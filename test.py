import torch
import torch.optim as optim
import torch_mimicry as mmc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'stl10_48', 'lsun_bedroom_128'])
parser.add_argument('--log_dir', type=str, default='./log/cifar100')
parser.add_argument('--step', type=int, default=100000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_runs', type=int, default=1)
opt = parser.parse_args()

# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Define models and optimizers
if opt.dataset == 'cifar100':
    import models.ssd_sngan_32 as ssd_sngan
    netG = ssd_sngan.SSD_SNGANGenerator32().to(device)
elif opt.dataset == 'stl10_48':
    import models.ssd_sngan_48 as ssd_sngan
    netG = ssd_sngan.SSD_SNGANGenerator48().to(device)
elif opt.dataset == 'lsun_bedroom_128':
    import models.ssd_sngan_128 as ssd_sngan
    netG = ssd_sngan.SSD_SNGANGenerator128().to(device)

# Evaluate fid
mmc.metrics.evaluate(
    metric='fid',
    log_dir=opt.log_dir,
    netG=netG,
    dataset_name=opt.dataset,
    num_real_samples=50000,
    num_fake_samples=50000,
    evaluate_step=opt.step,
    start_seed=opt.seed,
    num_runs=opt.num_runs,
    device=device)