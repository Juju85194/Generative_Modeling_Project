import torchvision
from PIL import Image
import torch
from data.guided_diffusion.unet import create_model
import matplotlib.pyplot as plt


def pilimg_to_tensor(pil_img, device="cuda"):
    t = torchvision.transforms.ToTensor()(pil_img)
    t = 2*t-1 # [0,1]->[-1,1]
    t = t.unsqueeze(0)
    t = t.to(device)
    return(t)


def display_as_pilimg(t, show=True):
    t = 0.5+0.5*t.to('cpu')
    t = t.squeeze()
    t = t.clamp(0.,1.)
    pil_img = torchvision.transforms.ToPILImage()(t)
    if show:
        display(pil_img)
    return(pil_img)


def gaussian_kernel(size, sigma, device="cuda"):
    """ 1D Gaussian Kernel """
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def init_model(device="cuda"):
    model_config = {'image_size': 256,
                    'num_channels': 128,
                    'num_res_blocks': 1,
                    'channel_mult': '',
                    'learn_sigma': True,
                    'class_cond': False,
                    'use_checkpoint': False,
                    'attention_resolutions': 16,
                    'num_heads': 4,
                    'num_head_channels': 64,
                    'num_heads_upsample': -1,
                    'use_scale_shift_norm': True,
                    'dropout': 0.0,
                    'resblock_updown': True,
                    'use_fp16': False,
                    'use_new_attention_order': False,
                    'model_path': '../data/ffhq_10m.pt'}
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    return model 


def display_logs(tensor_dict, save_path=False):
    num_rows = len(tensor_dict)
    num_cols = max(len(v) for v in tensor_dict.values())

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    for i, (idx, tensors) in enumerate(tensor_dict.items()):
        for j, tensor in enumerate(tensors):
            img = display_as_pilimg(tensor, show=False)
            axes[i][j].imshow(img)
            axes[i][j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
