import torch
import numpy as np

from .utils import display_as_pilimg

from PIL import Image
from tqdm import tqdm



class DDPM:
  def __init__(self, model=None, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=500, device="cuda"):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.device = device
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)


  def get_eps_from_model(self, x, t):
    # self.model returns:
    # - eps [0:3]
    # - variance [3:5]
    model_output = self.model(x, torch.tensor(t, device=self.device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.) 
    return(x_start)

  def sample(self, show_steps=True):
    with torch.no_grad():
      x = torch.randn(self.imgshape, device=self.device)  # initialize x_t for t=T
      for i, t in enumerate(self.reversed_time_steps):

          if t > 0:
            z = torch.randn(self.imgshape, device=self.device)
          else:
            z = z = torch.zeros_like(x)

          eps = self.get_eps_from_model(x, t)
          n = np.sqrt(self.betas[t])*z
          x = 1/np.sqrt(self.alphas[t])*(x-(1-self.alphas[t])/np.sqrt(1-self.alphas_cumprod[t])*eps)+n

          if i%100==0:
            print('Iteration :', t+1)
            pilimg = display_as_pilimg(torch.cat((x, self.predict_xstart_from_eps(x,eps,t)), dim=3))

    return(x)

  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None,
                         steps_viz=200):
    if vis_y==None:
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

      gradterm = torch.sum((linear_operator(xhat) - y)**2)
      grad = torch.autograd.grad(gradterm, x)[0]
      zeta = 1 / torch.sqrt(gradterm)

      x = x_prime - zeta*grad
      xt_s.append(x.detach())
      x0_s.append(xhat.detach().cpu())
      if show_steps and t%steps_viz==0:
        print('Iteration :', t)
        _ = display_as_pilimg(torch.cat((x, xhat, y, x_true), dim=3))
    
    if show_steps:
      _ = display_as_pilimg(torch.cat((x, xhat, y, x_true), dim=3))
    return (list(reversed(xt_s)), list(reversed(x0_s)))
  