import torch
import numpy as np

from src.utils import display_as_pilimg

from PIL import Image
from tqdm import tqdm


class DDIM:
    def __init__(self, model=None, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000, 
                 eta=0.2, device="cuda"):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device
        self.eta = eta

        self.betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]

        self.model = model
        self.imgshape = (1, 3, 256, 256)

    def get_eps_from_model(self, x, t):
        model_output = self.model(x, torch.tensor(t, device=self.device).unsqueeze(0))
        return model_output[:, :3, :, :]

    def predict_xstart_from_eps(self, x, eps, t):
        return (
            np.sqrt(1.0 / self.alphas_cumprod[t]) * x
            - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
        ).clamp(-1., 1.)

    def sample(self, show_steps=True):
        with torch.no_grad():
            x = torch.randn(self.imgshape, device=self.device)  # Initialisation de x_T
            for i in range(len(self.reversed_time_steps)-1):
                t = self.reversed_time_steps[i]
                t_next = self.reversed_time_steps[i + 1]

                eps = self.get_eps_from_model(x, t)
                x0_pred = self.predict_xstart_from_eps(x, eps, t)

                alpha_t = self.alphas_cumprod[t]
                alpha_t_next = self.alphas_cumprod[t_next]

                sigma_t = self.eta * np.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_t_next)
                noise = torch.randn_like(x) if self.eta > 0 else 0

                x = np.sqrt(alpha_t_next) * x0_pred + np.sqrt(1 - alpha_t_next - sigma_t**2) * eps + sigma_t * noise

                if show_steps and i % 10 == 0:
                    print(f'Iteration : {t}')
                    _ = display_as_pilimg(x)

        return x

    def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None, steps_viz=200):
        if vis_y is None:
            vis_y = y

        xt_s = []
        x0_s = []

        x = torch.randn(self.imgshape, device=self.device)
        x.requires_grad = True

        for i in range(len(self.reversed_time_steps) - 1):
            t = self.reversed_time_steps[i]
            t_next = self.reversed_time_steps[i + 1]

            eps = self.get_eps_from_model(x, t)
            xhat = self.predict_xstart_from_eps(x, eps, t)  # x_0

            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next]

            sigma_t = self.eta * np.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_t_next)
            noise = torch.randn_like(x) if self.eta > 0 else 0

            x_prime = np.sqrt(alpha_t_next) * xhat + np.sqrt(1 - alpha_t_next - sigma_t**2) * eps + sigma_t * noise

            gradterm = torch.sum((linear_operator(xhat) - y) ** 2)
            grad = torch.autograd.grad(gradterm, x)[0]
            zeta = 1 / torch.sqrt(gradterm) 

            x = x_prime - zeta * grad

            xt_s.append(x.detach())
            x0_s.append(xhat.detach().cpu())

            if show_steps and i % steps_viz == 0:
                print(f'Iteration : {t}')
                _ = display_as_pilimg(torch.cat((x, xhat, y, x_true), dim=3))

        if show_steps:
            _ = display_as_pilimg(torch.cat((x, xhat, y, x_true), dim=3))

        return list(reversed(xt_s)), list(reversed(x0_s))
