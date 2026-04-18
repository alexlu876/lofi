"""CycleGAN loss functions (spec §4).

Includes R1 gradient penalty (Mescheder et al., ICML 2018) and
one-sided label smoothing (Salimans et al., NeurIPS 2016).
"""

import random

import torch
import torch.nn as nn


class ImagePool:
    """Buffer of 50 most recent generated images for discriminator training."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img.detach().clone())
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                old = self.images[idx].clone()
                self.images[idx] = img.detach().clone()
                result.append(old)
            else:
                result.append(img)
        return torch.cat(result, dim=0)


def lsgan_loss_d(real_pred: torch.Tensor, fake_pred: torch.Tensor,
                 smooth_real: float = 0.9) -> torch.Tensor:
    """LSGAN discriminator loss with one-sided label smoothing.

    Real target is `smooth_real` (default 0.9) instead of 1.0.
    Fake target remains 0.0 (never smooth fake labels).
    """
    return torch.mean((real_pred - smooth_real) ** 2) + torch.mean(fake_pred ** 2)


def lsgan_loss_g(fake_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((fake_pred - 1.0) ** 2)


def r1_gradient_penalty(discriminator: nn.Module, real_input: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty (Mescheder et al., ICML 2018).

    Penalizes ||∇D(real)||² to prevent discriminator overconfidence.
    Runs on CPU if MPS to avoid autograd incompatibilities.
    """
    orig_device = real_input.device
    use_cpu = orig_device.type == "mps"

    if use_cpu:
        real_input = real_input.detach().cpu().requires_grad_(True)
        disc_cpu = discriminator.cpu()
        real_pred = disc_cpu(real_input)
    else:
        real_input = real_input.detach().requires_grad_(True)
        real_pred = discriminator(real_input)

    grad, = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_input,
        create_graph=True,
    )
    penalty = grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1).mean()

    if use_cpu:
        discriminator.to(orig_device)
        return penalty.to(orig_device)
    return penalty


def cycle_consistency_loss(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(reconstructed, original)


def identity_loss(same_domain_output: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(same_domain_output, real_input)
