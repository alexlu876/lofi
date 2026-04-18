"""HiFi-GAN loss functions (spec §5.4)."""

import torch


def discriminator_loss(disc_real_outputs: list, disc_gen_outputs: list):
    loss = 0.0
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        loss += torch.mean((dr - 1.0) ** 2) + torch.mean(dg ** 2)
    return loss


def generator_loss(disc_outputs: list):
    loss = 0.0
    for dg in disc_outputs:
        loss += torch.mean((dg - 1.0) ** 2)
    return loss


def feature_matching_loss(fmap_real: list, fmap_gen: list):
    loss = 0.0
    for fr_list, fg_list in zip(fmap_real, fmap_gen):
        for fr, fg in zip(fr_list, fg_list):
            loss += torch.mean(torch.abs(fr.detach() - fg))
    return loss
