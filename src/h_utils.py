import torch
import torch.nn.functional as F


def inpainting_h(x, mask):
    """Applies the inpainting mask."""
    return x * mask


def inpainting_h_pinv(x, mask):
    """Pseudoinverse for inpainting (same as h)."""
    return x * mask


def superres_h(x, scale_factor=4):
    """Downsamples the image (average pooling)."""
    return F.avg_pool2d(x, kernel_size=scale_factor)


def superres_h_pinv(x_lowres, scale_factor=4, mode='bicubic'):
    """Upsamples the image using bicubic interpolation."""
    return F.interpolate(x_lowres, scale_factor=scale_factor, mode=mode, align_corners=False)