import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
from PIL import Image
from src.utils import *
from src.h_fcn import *
from src.pigdm import PIGDM

def calculate_psnr(img_pred, img_true, device):
    """Calculates PSNR between predicted and true images."""
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device)
    img_pred = (0.5 + 0.5 * img_pred).clamp(0.0, 1.0)  # Scale to [0, 1]
    img_true = (0.5 + 0.5 * img_true).clamp(0.0, 1.0)
    return psnr(img_pred, img_true)

def calculate_ssim(img_pred, img_true, device):
    """Calculates SSIM between predicted and true images."""
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    img_pred = (0.5 + 0.5 * img_pred).clamp(0.0, 1.0)  # Scale to [0, 1]
    img_true = (0.5 + 0.5 * img_true).clamp(0.0, 1.0)
    return ssim(img_pred, img_true)

def calculate_lpips(img_pred, img_true, device):
    """Calculates LPIPS between predicted and true images."""
    lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='mean').to(device)
    img_pred = (0.5 + 0.5 * img_pred).clamp(0.0, 1.0)  # Scale to [0, 1]
    img_true = (0.5 + 0.5 * img_true).clamp(0.0, 1.0)
    return lpips(img_pred, img_true)

def evaluate_model(model, h_fcn_class, dataset_path, num_images=10, num_samples=1, eta=1.0, awd=True, show_steps=False, steps_viz=200, device="cuda", **kwargs):
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for idx in tqdm(range(num_images), desc="Evaluating Images"):
        img_pil = Image.open(f'{dataset_path}/{str(idx).zfill(5)}.png')
        x_true = pilimg_to_tensor(img_pil)

        if h_fcn_class == Inpainting:
            mask = kwargs.get('mask', None)
            if mask is None:
                mask = torch.ones(imgshape, device=device)
                h, w = 256, 256
                hcrop, wcrop = h // 4, w // 4
                corner_top, corner_left = int(h // 1.7) - hcrop // 1, w // 2 - wcrop // 2
                mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0
            H = h_fcn_class(mask)
        elif h_fcn_class == Superres:
            scale_factor = kwargs.get('scale_factor', 4)
            H = h_fcn_class(scale_factor)
        elif h_fcn_class == Blur:
            kernel_size = kwargs.get('kernel_size', 9)
            sigma = kwargs.get('sigma', 3.0)
            H = h_fcn_class(kernel_size, sigma)
        else:
            raise ValueError("Unsupported h_fcn_class")

        y = H(x_true)
        y_0 = y.repeat(num_samples, 1, 1, 1)
        pigdm_instance = PIGDM(model, 256, H)

        for _ in range(num_samples):
            xt_s, _ = pigdm_instance.posterior_sampling(y=y_0, x_true=x_true, eta=eta, awd=awd, show_steps=show_steps, steps_viz=steps_viz)
            x_pred = xt_s[0]

            psnr_val = calculate_psnr(x_pred, x_true, device).item()
            ssim_val = calculate_ssim(x_pred, x_true, device).item()
            lpips_val = calculate_lpips(x_pred, x_true, device).item()

            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_lpips.append(lpips_val)

    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_lpips = np.mean(all_lpips)

    return {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}