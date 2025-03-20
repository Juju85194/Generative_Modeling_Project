import torch
import numpy as np

from .utils import display_as_pilimg

from PIL import Image
from tqdm import tqdm

from .ddpm import DDPM


class PIGDM(DDPM):
    def __init__(self, model=None, n=256, H=None, sigma_y=0, eta=0.5, num_diffusion_timesteps=1000, device="cuda"):
        super().__init__(model, num_diffusion_timesteps=num_diffusion_timesteps, device=device)
        self.H = H
        self.sigma_y = sigma_y
        self.noisy = True if sigma_y !=0 else False
        self.eta = eta  # Keep this for internal default
        self.sigma_t = torch.sqrt((1 - torch.tensor(self.alphas_cumprod, dtype=torch.float32)) / torch.tensor(self.alphas_cumprod, dtype=torch.float32)).to(device)
        self.n = n
        self.device = device
        self.sigma_y = sigma_y

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

    def _get_grad_term(self, x_t, xhat, y, t=1):
        if not self.noisy:
            mat_term = (self.H.H_pinv(y) - self.H.H_pinv(self.H.H(xhat))).reshape(self.n, -1)
            mat_x = (mat_term.detach() * xhat.reshape(self.n, -1)).sum()
            grad_term = torch.autograd.grad(mat_x, x_t, retain_graph=True)[0].detach()
        else:
            r = self.sigma_t[t] / (1 + self.sigma_t[t])
            mat_term = (y - self.H(xhat)) @ torch.inverse((self.H.mat @ self.H.mat.T + self.sigma_y / r * torch.eye(self.n).to(self.device))) @ self.H.mat######
            mat_x = (mat_term.detach().reshape(self.n, -1) * xhat.reshape(self.n, -1)).sum()
            grad_term = torch.autograd.grad(mat_x, x_t, retain_graph=True)[0].detach()            

        return grad_term

    def _init_sampling(self):
        xt_s = []
        x0_s = []

        self.alphas_cp_torch = torch.tensor(self.alphas_cumprod, device=self.device, dtype=torch.float32)

        x = torch.randn(self.imgshape, device=self.device)
        x.requires_grad = True

        t_s = self.reversed_time_steps
        v_s = [-1] + list(t_s[::-1][:-1])
        v_s = v_s[::-1]

        return (x, xt_s, x0_s, x, t_s, v_s)

    def _get_coefs(self, x, ti, si, awd):
        alpha_t = self.alphas_cp_torch[ti].view(-1, 1, 1, 1)
        alpha_s = torch.tensor(self.alphas_cp_torch[si].cpu() if si != -1 else 1.0,
                               device=self.device,
                               dtype=torch.float32).view(-1, 1, 1, 1)

        c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
        c2 = ((1 - alpha_s) - c1**2).sqrt()

        if awd:
            coef_g = alpha_t.sqrt() * alpha_s.sqrt()
        else:
            coef_g = (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt()) * alpha_t.sqrt()
        return (alpha_t, alpha_s, c1, c2, coef_g)

    def ddpm_sampling(self, y, x_true=None, show_steps=True, vis_y=None,
                      steps_viz=200, eta=None, beta=1, awd=True, tqdm_bar=True):
        if vis_y is None:
            vis_y = y                

        xt_s = []
        x0_s = []

        x = torch.randn(self.imgshape,device=self.device)
        x.requires_grad = True

        for t in self.reversed_time_steps:
            z = torch.randn(self.imgshape, device=self.device)
            eps = self.get_eps_from_model(x, t)

            xhat = self.predict_xstart_from_eps(x, eps, t)

            x_prime = np.sqrt(self.alphas[t])*(1-self.alphas_cumprod_prev[t])/(1-self.alphas_cumprod[t])*x
            x_prime += np.sqrt(self.alphas_cumprod_prev[t])*self.betas[t]/(1-self.alphas_cumprod[t])*xhat + np.sqrt(self.betas[t])*z
            grad_term = self._get_grad_term(x_prime, xhat, y, t)

            x = x_prime - zeta*grad
            xt_s.append(x.detach())
            x0_s.append(xhat.detach().cpu())
            if show_steps and t%100==0:
                print('Iteration :', t)
                _ = display_as_pilimg(torch.cat((x, xhat, y, x_true), dim=3))
            
    def posterior_sampling(self, y, x_true=None, show_steps=True, vis_y=None,
                           steps_viz=200, eta=None, beta=1, awd=True, tqdm_bar=True):
        """
        apply ddim sampling
        y: the input image
        x_true: the true image (optionnal)
        show_steps: verbose
        vis_y: not useful #todo delete it
        eta: parameter of the ddim sampler
        beta: grad term weight
        awd: adaptative weight diffusion True or false
        """
        if vis_y is None:
            vis_y = y

        if eta is None:
            eta = self.eta

        y = y.repeat(1,1,1,1)

        (x, xt_s, x0_s, x, t_s, v_s) = self._init_sampling()

        x_t = x
        if tqdm_bar:
            time_iterator = tqdm(zip(t_s, v_s), total=len(t_s), desc="Sampling")
        else:
            time_iterator = zip(t_s, v_s)

        for i, (t_, s_) in enumerate(time_iterator):
            alpha_t, alpha_s, c1, c2, coef_g = self._get_coefs(x, t_, s_, awd)
            x_t = x_t.clone().to(self.device).requires_grad_(True)
            eps = self.get_eps_from_model(x_t, t_)
            xhat = self.predict_xstart_from_eps(x_t, eps, t_)
            grad_term = self._get_grad_term(x_t, xhat, y, t_)
            xhat = xhat.detach()
            eps = eps.detach()
            z = torch.randn(self.imgshape, device=self.device)

            if not awd:
                eps = (x_t - xhat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()

            x_s = alpha_s.sqrt() * xhat + c1 * z + c2 * eps + coef_g * grad_term * beta

            xt_s.append(x_s.detach())
            x0_s.append(xhat.detach().cpu())
            x_t = x_s.detach()
            x_t.requires_grad_(False)

            if show_steps and i % steps_viz == 0:
                self._show_step(xhat, y, x_true, x_t)

        if show_steps:
            self._show_step(xhat, y, x_true, x_t)

        return (list(reversed(xt_s)), list(reversed(x0_s)))

    def _show_step(self, xhat, y, x_true, x_t=None):
        tensors = [x_t, xhat, self.H.show(y), x_true]
        tensors = [tens.cpu() for tens in tensors if tens is not None]

        _ = display_as_pilimg(torch.cat(tensors, dim=3))
