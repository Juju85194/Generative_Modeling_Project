import torch
import torch.nn.functional as F


def inpainting_h(x, mask):
    """Applies the inpainting mask."""
    return x * mask


def inpainting_h_pinv(x, mask):
    """Pseudoinverse for inpainting (same as h)."""
    return x


def superres_h(x, scale_factor=4):
    """Downsamples the image (average pooling)."""
    return F.avg_pool2d(x, kernel_size=scale_factor)


def superres_h_pinv(x_lowres, scale_factor=4, mode='bicubic'):
    """Upsamples the image using bicubic interpolation."""
    return F.interpolate(x_lowres, scale_factor=scale_factor, mode=mode, align_corners=False)


def matmul(x, mat):
    """Applies matrice multiplication."""
    return mat @ x


def mat_pinv(x, mat):
    """return p_inv @ x of the matrix """
    p_inv = torch.linalg.pinv(mat)
    return p_inv @ x


def compute_svd(kernel, img_dim, device):
    H_small = torch.zeros(img_dim, img_dim, device=device)
    kernel_size = kernel.shape[0]

    for i in range(img_dim):
        for j in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
            if 0 <= j < img_dim:
                H_small[i, j] = kernel[j - i + kernel_size // 2]

    U_small, singulars_small, V_small = torch.svd(H_small, some=False)
    singulars_small[singulars_small < 3e-2] = 0

    return U_small, singulars_small, V_small


def apply_matrix(M, vec, img_dim):
    batch_size = vec.shape[0]
    vec = vec.view(batch_size * 3, img_dim, img_dim)
    vec = torch.matmul(M, vec)
    vec = torch.matmul(vec, M.T)
    return vec.view(batch_size, 3, img_dim, img_dim) # Output same shape as input


def blur(x, kernel, img_dim=256, device="cuda"):
    """
    Its called blur but actually its just a function to apply a 1D kernel
    """
    U_small, singulars_small, V_small = compute_svd(kernel, img_dim, device)
    temp = apply_matrix(V_small.T, x, img_dim)
    temp = singulars_small.view(1, 1, img_dim, 1) * temp
    return apply_matrix(U_small, temp, img_dim)


def blur_inv(x, kernel, img_dim=256, device="cuda"):
    """
    blur_inv but its not inv but pinv
    """
    U_small, singulars_small, V_small = compute_svd(kernel, img_dim, device)
    temp = apply_matrix(U_small.T, x, img_dim)
    nonzero_idx = singulars_small.nonzero().flatten()
    singulars_inv = torch.zeros_like(singulars_small)
    singulars_inv[nonzero_idx] = 1 / singulars_small[nonzero_idx]
    temp = singulars_inv.view(1, 1, img_dim, 1) * temp
    return apply_matrix(V_small, temp, img_dim)
