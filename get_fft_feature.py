import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms


def RGB2gray(rgb):
    if rgb.size(1) == 3:
        r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    elif rgb.size(1) == 1:
        return rgb[:,0,:,:]

def shift(x):
    out = torch.zeros(x.size())
    H, W = x.size(-2), x.size(-1)
    out[:,:int(H/2),:int(W/2)] = x[:,int(H/2):,int(W/2):]
    out[:,:int(H/2),int(W/2):] = x[:,int(H/2):,:int(W/2)]
    out[:,int(H/2):,:int(W/2)] = x[:,:int(H/2),int(W/2):]
    out[:,int(H/2):,int(W/2):] = x[:,:int(H/2),:int(W/2)]
    return out

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = np.indices([H, W])
    radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)
    radius = radius.astype(np.int).ravel()
    nr = np.bincount(radius)
    tbin = np.bincount(radius, image.ravel())
    radial_prof = tbin / (nr + 1e-10)
    return radial_prof[1:-2] # We ignore the last two extremely high frequency bands to avoid noise.

def get_fft_feature(x):
    x_rgb = x.detach()
    epsilon = 1e-8

    x_gray = RGB2gray(x_rgb)
    fft = torch.rfft(x_gray,2,onesided=False)
    fft += epsilon
    magnitude_spectrum = torch.log((torch.sqrt(fft[:,:,:,0]**2 + fft[:,:,:,1]**2 + 1e-10))+1e-10)
    magnitude_spectrum = shift(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum.cpu().numpy()

    out = []
    for i in range(magnitude_spectrum.shape[0]):
        out.append(torch.from_numpy(azimuthalAverage(magnitude_spectrum[i])).float().unsqueeze(0))
    out = torch.cat(out, dim=0)
    out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (torch.max(out, dim=1, keepdim=True)[0] - torch.min(out, dim=1, keepdim=True)[0])
    out = Variable(out, requires_grad=True).to(x.device)
    return out