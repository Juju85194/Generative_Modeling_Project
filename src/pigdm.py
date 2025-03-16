import torch
import numpy as np

from .utils import display_as_pilimg

from PIL import Image
from tqdm import tqdm

from .ddpm import DDPM


class PIGDM(DDPM):
    def __init__(self, model=None, n=256, H=None, sigma_y=0, eta=0.5, device="cuda"):
        super().__init__(model, device=device)
        self.H = H
        self.sigma_y = sigma_y
        self.noisy = True if sigma_y !=0 else False
        self.eta = eta
        self.sigma_t = np.linspace(0.001, 1) ### temp
        self.n = n
        self.device = "cuda"

    def _get_rt(self):
        if self.noisy:
            return np.sqrt(self.sigma_t/(self.sigma_t+1))
        else:
            return None ### TODO

    def get_noisy_xt(self, x0, t, noise=None):
        """Adds noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = np.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alphas_cumprod[t])
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt

    def init_x(self, x, y, t, y_0):
        H = self.H
        n = x.size(0)
        x_0 = self.H.H_pinv(y_0).view(*x.size()).detach()
        t_s = torch.ones(self.n).to(x.device).long() * t
        alpha_t = self.alphas[t_s].view(-1, 1, 1, 1)
        return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)

    def _get_grad_term(self, x_t, xhat, y):
        if not self.noisy:
            mat_term = (self.H.H_pinv(y) - self.H.H_pinv(self.H.H(xhat))).reshape(self.n, -1)

            mat_x = (mat_term.detach() * xhat.reshape(self.n, -1)).sum()

            grad_term = torch.autograd.grad(mat_x, x_t, retain_graph=True)[0].detach()
        else:
            mat_term = (y_0 - self.H(xhat)).T @ (self.H.val @ self.H.val.T + self.sigma_y/self.r_t)######
            raise ValueError("Not implemented yet")
        
        return grad_term

    def _init_sampling(self):
        xt_s = []
        x0_s = []

        self.alphas_cp_torch = torch.tensor(self.alphas_cumprod, device=self.device, dtype=torch.float32)

        x = torch.randn(self.imgshape, device=self.device)
        x.requires_grad = True

        t_s = self.reversed_time_steps
        v_s = [-1] + t_s[::-1][:-1] # in the pseudo code it's v_i's the time step and they note t and s
        # here i note t_s the list for the t's and v_s the list for the s's
        v_s = v_s[::-1]

        return (x, xt_s, x0_s, x, t_s, v_s)

    def _get_coefs(self, x, ti, si):
        t = torch.ones(self.n).to(x.device).long() * ti
        s = torch.ones(self.n).to(x.device).long() * si

        alpha_t = self.alphas_cp_torch[ti].view(-1, 1, 1, 1)
        alpha_s = torch.tensor(self.alphas_cp_torch[si].cpu() if si != -1 else 1.0, device=self.device, dtype=torch.float32).view(-1, 1, 1, 1)

        c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
        c2 = ((1 - alpha_s) - c1**2).sqrt()

        coef_g = alpha_t.sqrt() * alpha_s.sqrt()
        return (alpha_t, alpha_s, c1, c2, coef_g)

    def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None, steps_viz=200):

        if vis_y == None:
            vis_y = y

        y = y.repeat(1, 1, 1, 1)

        (x, xt_s, x0_s, x, t_s, v_s) = self._init_sampling()

        x_t = x

        for i, (t_, s_) in enumerate(tqdm(zip(t_s, v_s), total=len(t_s))):

            alpha_t, alpha_s, c1, c2, coef_g = self._get_coefs(x, t_, s_)
            x_t = x_t.clone().to(self.device).requires_grad_(True)

            eps = self.get_eps_from_model(x_t, t_)

            xhat = self.predict_xstart_from_eps(x_t, eps, t_)

            grad_term = self._get_grad_term(x_t, xhat, y)

            xhat = xhat.detach()
            eps = eps.detach()

            z = torch.randn(self.imgshape, device=self.device) # eps in the pseudo code but eps was already used :(
            x_s = alpha_s.sqrt() * xhat + c1 * z + c2 * eps + coef_g * grad_term

            xt_s.append(x_s.detach().cpu())
            x0_s.append(xhat.detach().cpu())
            x_t = x_s.detach()
            x_t.requires_grad_(False)

            if show_steps and i % steps_viz == 0:
                _ = display_as_pilimg(torch.cat((x_t, xhat, self.H.show(y), x_true), dim=3))
        return (list(reversed(xt_s)), list(reversed(x0_s)))
