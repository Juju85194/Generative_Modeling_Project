import torch
import torch.nn.functional as F
from .h_utils import *
from .utils import gaussian_kernel


class H_fcn:
    def __init__(self, H=None, H_pinv=None, values=None, linear=True):
        self.H = H
        self.linear = linear
        self.val = values
        self.H_pinv = H_pinv

    def __call__(self, x):
        return self.H(x)

    def show(self, x):
        return self.H(x)


class Inpainting(H_fcn):
    def __init__(self, mask):
        self.H = lambda x: inpainting_h(x, mask)
        self.H_pinv = lambda x: inpainting_h_pinv(x, mask)
        self.superres = False
        super().__init__(self.H, self.H_pinv, None, False)


class Superres(H_fcn):
    def __init__(self, factor=4):
        self.H = lambda x: superres_h(x, factor)
        self.H_pinv = lambda x: superres_h_pinv(x, factor)
        self.superres = True
        super().__init__(self.H, self.H_pinv, None, False)

    def show(self, x, img_size=(256, 256)):
        return F.interpolate(self(x), size=img_size, mode='bicubic', align_corners=False)


class Linear(H_fcn):
    def __init__(self, mat=torch.ones(1, 3, 256, 256)):
        self.H = lambda x: matmul(x, mat)
        self.H_pinv = lambda x: mat_pinv(x, mat)
        self.superres = False
        self.mat = mat
        super().__init__(self.H, self.H_pinv, mat, True)


class Blurring(H_fcn):
    def __init__(self, kernel="Gaussian", size=11, sigma=5, device="cuda"):
        ### for size use an odd number
        if kernel == "Gaussian":
            self.kernel = gaussian_kernel(size, sigma, device)
        else:
            self.kernel = kernel

        self.H = lambda x: blur(x, self.kernel, img_dim=256, device=device)
        self.H_pinv = lambda x: blur_inv(x, self.kernel, img_dim=256, device=device)
        self.superres = False
        self.mat = None
        super().__init__(self.H, self.H_pinv, False, False)
