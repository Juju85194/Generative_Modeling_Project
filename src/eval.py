import torch
import os
import numpy as np
import random
import torchmetrics
from tqdm import tqdm
from PIL import Image
from src.utils import *
from src.h_fcn import *
from src.pigdm import PIGDM
from src.ddpm import DDPM
from src.ddim import DDIM

IMGSHAPE = (1, 3, 256, 256)

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

def calculate_fid(img_pred, img_true, device):
    """Calculates FID between predicted and true images."""
    fid = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device) # Use normalize=True

    # Ensure images are in the [0, 255] range and uint8 type
    img_pred = (img_pred.clamp(0.0, 1.0) * 255).type(torch.uint8)
    img_true = (img_true.clamp(0.0, 1.0) * 255).type(torch.uint8)
    
    fid.update(img_true, real=True)
    fid.update(img_pred, real=False)
    return fid.compute()

def evaluate_model(model, h_fcn_class, dataset_path, num_images=10,
                   num_samples=1, eta=1.0, awd=True, show_steps=False,
                   num_diffusion_timesteps=500, num_ddim_steps=500, steps_viz=200,
                   device="cuda", keep_logs=False, run_dps=False, **kwargs):
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_fid = []

    if run_dps:
        all_psnr_d = []
        all_ssim_d = []
        all_lpips_d = []
        all_fid_d = []
        all_logs_d = {}

    all_logs = {}

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize FID metric
    fid = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    if run_dps:
        fid_d = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for idx in tqdm(range(num_images), desc="Evaluating Images"):
        selected_img = random.choice(image_files)
        img_pil = Image.open(os.path.join(dataset_path, selected_img))
        x_true = pilimg_to_tensor(img_pil)

        all_logs[idx] = [x_true.cpu()]

        if run_dps:
            all_logs_d[idx] = [x_true.cpu()]

        if h_fcn_class == Inpainting:
            mask = kwargs.get('mask', None)
            if mask is None:
                mask = torch.ones(IMGSHAPE, device=device)
                h, w = 256, 256
                hcrop, wcrop = h // 4, w // 4
                corner_top, corner_left = int(h // 1.7) - hcrop // 1, w // 2 - wcrop // 2
                mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0
            H = h_fcn_class(mask.to(device))
        
        elif h_fcn_class == Linear:
            base_mat = torch.eye(256).to(device)
            base_mat[90:130] = 0
            mat = kwargs.get("mask", base_mat)
            H = h_fcn_class(mat.to(device))

        elif h_fcn_class == Superres:
            scale_factor = kwargs.get('scale_factor', 4)
            H = h_fcn_class(scale_factor)

        elif h_fcn_class == Blurring:
            kernel_size = kwargs.get('kernel_size', 13)
            sigma = kwargs.get('sigma', 5.0)

            kernel = gaussian_kernel(size=kernel_size, sigma=sigma, device=device)

            H = h_fcn_class(kernel=kernel)

        else:
            raise ValueError("Unsupported h_fcn_class")

        y = H(x_true)
        all_logs[idx].append(H.show(x_true).cpu())

        if run_dps:
            all_logs_d[idx].append(H.show(x_true).cpu())

        y_0 = y#.repeat(num_samples, 1, 1, 1)
        pigdm_instance = PIGDM(model, 256, H, num_diffusion_timesteps=num_diffusion_timesteps,
                               num_ddim_steps=num_ddim_steps)

        if run_dps:
            dps_instance = DDIM(model, num_diffusion_timesteps=num_diffusion_timesteps,
                                num_ddim_steps=num_ddim_steps)

        for _ in range(num_samples):
            xt_s, _ = pigdm_instance.posterior_sampling(y=y_0, x_true=x_true, eta=eta, awd=awd, show_steps=show_steps, steps_viz=steps_viz,
                                                        tqdm_bar=False)

            x_pred = xt_s[0]

            fid.update((x_true.clamp(0.0, 1.0) * 255).type(torch.uint8).to(device), real=True)
            fid.update((x_pred.clamp(0.0, 1.0) * 255).type(torch.uint8).to(device), real=False)


            psnr_val = calculate_psnr(x_pred, x_true, device).item()
            ssim_val = calculate_ssim(x_pred, x_true, device).item()
            lpips_val = calculate_lpips(x_pred, x_true, device).item()


            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_lpips.append(lpips_val)

            all_logs[idx].append(x_pred.cpu())

            if run_dps:

                xt_s_d, _ = dps_instance.posterior_sampling(H.show, y=H.show(x_true), x_true=x_true,
                                                            show_steps=show_steps, eta=eta,
                                                            steps_viz=steps_viz)
                x_pred_d = xt_s_d[0]

                fid_d.update((x_true.clamp(0.0, 1.0) * 255).type(torch.uint8).to(device), real=True)
                fid_d.update((x_pred_d.clamp(0.0, 1.0) * 255).type(torch.uint8).to(device), real=False)

                psnr_val_d = calculate_psnr(x_pred_d, x_true, device).item()
                ssim_val_d = calculate_ssim(x_pred_d, x_true, device).item()
                lpips_val_d = calculate_lpips(x_pred_d, x_true, device).item()

                all_psnr_d.append(psnr_val_d)
                all_ssim_d.append(ssim_val_d)
                all_lpips_d.append(lpips_val_d)

                all_logs_d[idx].append(x_pred_d.cpu())

    avg_fid = fid.compute().item()
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_lpips = np.mean(all_lpips)

    scores = {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips, "FID": avg_fid}

    if run_dps:
        avg_fid_d = fid_d.compute().item()
        scores["PSNR_d"] = np.mean(all_psnr_d)
        scores["SSIM_d"] = np.mean(all_ssim_d)
        scores["LPIPS_d"] = np.mean(all_lpips_d)
        scores["FID_d"] = avg_fid_d


    if run_dps:
        return (scores, (all_logs, all_logs_d))

    return (scores, all_logs)
