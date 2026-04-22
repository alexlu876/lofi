"""Phase 1: CycleGAN training on mel spectrograms (spec §6, Phase 1)."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.mel_dataset import MelDataset
from src.losses.cyclegan_losses import (
    ImagePool,
    cycle_consistency_loss,
    identity_loss,
    lsgan_loss_d,
    lsgan_loss_g,
)
from src.models.cyclegan_generator import UNetGenerator
from src.models.ema import EMA
from src.models.patchgan import PatchGANDiscriminator


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def get_lr_lambda(total_epochs: int, decay_start: int):
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / (total_epochs - decay_start)
    return lr_lambda


def validate(G, F_net, val_clean_loader, val_lofi_loader, device, amp_dtype, use_bf16):
    G.eval()
    F_net.eval()
    total_cyc = 0.0
    n = 0
    with torch.no_grad():
        clean_iter = iter(val_clean_loader)
        lofi_iter = iter(val_lofi_loader)
        for _ in range(min(len(val_clean_loader), len(val_lofi_loader))):
            try:
                real_c = next(clean_iter).to(device)
                real_l = next(lofi_iter).to(device)
            except StopIteration:
                break
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_bf16):
                rec_c = F_net(G(real_c))
                rec_l = G(F_net(real_l))
                loss_cyc = (nn.functional.l1_loss(rec_c, real_c) +
                            nn.functional.l1_loss(rec_l, real_l))
            total_cyc += loss_cyc.item()
            n += 1
    G.train()
    F_net.train()
    return total_cyc / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-dir", default="logs/tensorboard/cyclegan")
    parser.add_argument("--ckpt-dir", default="checkpoints/cyclegan")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["cyclegan"]["training"]
    lc = cfg["cyclegan"]["loss"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    clean_ds = MelDataset.from_split_file(
        data_dir / "splits" / "clean_train.txt",
        data_dir / "stats" / "clean_mel_stats.json",
    )
    lofi_ds = MelDataset.from_split_file(
        data_dir / "splits" / "lofi_train.txt",
        data_dir / "stats" / "lofi_mel_stats.json",
    )
    clean_loader = DataLoader(clean_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    lofi_loader = DataLoader(lofi_ds, batch_size=tc["batch_size"], shuffle=True,
                             num_workers=4, pin_memory=(device == "cuda"), drop_last=True)

    val_clean_ds = MelDataset.from_split_file(
        data_dir / "splits" / "clean_val.txt",
        data_dir / "stats" / "clean_mel_stats.json",
    )
    val_lofi_ds = MelDataset.from_split_file(
        data_dir / "splits" / "lofi_val.txt",
        data_dir / "stats" / "lofi_mel_stats.json",
    )
    val_clean_loader = DataLoader(val_clean_ds, batch_size=tc["batch_size"], shuffle=False,
                                  num_workers=2, pin_memory=(device == "cuda"), drop_last=True)
    val_lofi_loader = DataLoader(val_lofi_ds, batch_size=tc["batch_size"], shuffle=False,
                                 num_workers=2, pin_memory=(device == "cuda"), drop_last=True)

    G = UNetGenerator().to(device)  # clean -> lofi
    F_net = UNetGenerator().to(device)  # lofi -> clean
    D_L = PatchGANDiscriminator().to(device)
    D_C = PatchGANDiscriminator().to(device)

    G.apply(init_weights)
    F_net.apply(init_weights)
    D_L.apply(init_weights)
    D_C.apply(init_weights)

    # Apply spectral norm on CUDA; weight clipping on MPS (spectral norm crashes MPS)
    use_weight_clip = device != "cuda"
    if device == "cuda":
        D_L.apply_spectral_norm()
        D_C.apply_spectral_norm()
        print("Using spectral norm on discriminators")
    else:
        print("Using weight clipping on discriminators (MPS fallback)")

    ema_G = EMA(G, decay=tc["ema_decay"])
    ema_F = EMA(F_net, decay=tc["ema_decay"])

    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(F_net.parameters()),
        lr=tc["lr"], betas=tuple(tc["betas"]),
    )
    opt_D = torch.optim.Adam(
        list(D_L.parameters()) + list(D_C.parameters()),
        lr=tc["lr"], betas=tuple(tc["betas"]),  # Same lr as G (original CycleGAN)
    )

    lr_lambda = get_lr_lambda(tc["total_epochs"], tc["lr_decay_start_epoch"])
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    pool_L = ImagePool(tc["image_pool_size"])
    pool_C = ImagePool(tc["image_pool_size"])

    use_bf16 = tc["precision"] == "bf16" and device == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    writer = SummaryWriter(args.log_dir)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        F_net.load_state_dict(ckpt["F"])
        D_L.load_state_dict(ckpt["D_L"])
        D_C.load_state_dict(ckpt["D_C"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        ema_G.load_state_dict(ckpt["ema_G"])
        ema_F.load_state_dict(ckpt["ema_F"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    lambda_cyc = lc["lambda_cyc"]
    lambda_id = lc["lambda_id"]
    global_step = start_epoch * min(len(clean_loader), len(lofi_loader))
    best_val_loss = float("inf")

    for epoch in range(start_epoch, tc["total_epochs"]):
        G.train()
        F_net.train()
        D_L.train()
        D_C.train()

        clean_iter = iter(clean_loader)
        lofi_iter = iter(lofi_loader)
        n_steps = min(len(clean_loader), len(lofi_loader))

        for step in range(n_steps):
            try:
                real_c = next(clean_iter).to(device)
                real_l = next(lofi_iter).to(device)
            except StopIteration:
                break

            # --- Generator step ---
            opt_G.zero_grad()
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_bf16):
                fake_l = G(real_c)
                fake_c = F_net(real_l)

                rec_c = F_net(fake_l)
                rec_l = G(fake_c)

                id_l = G(real_l)
                id_c = F_net(real_c)

                loss_adv_G = lsgan_loss_g(D_L(fake_l))
                loss_adv_F = lsgan_loss_g(D_C(fake_c))
                loss_cyc = cycle_consistency_loss(rec_c, real_c) + cycle_consistency_loss(rec_l, real_l)
                loss_id = identity_loss(id_l, real_l) + identity_loss(id_c, real_c)

                loss_G_total = loss_adv_G + loss_adv_F + lambda_cyc * loss_cyc + lambda_id * loss_id

            loss_G_total.backward()
            opt_G.step()

            ema_G.update(G)
            ema_F.update(F_net)

            # --- Discriminator step ---
            opt_D.zero_grad()
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_bf16):
                fake_l_pool = pool_L.query(fake_l.detach())
                fake_c_pool = pool_C.query(fake_c.detach())

                loss_D_L = lsgan_loss_d(D_L(real_l), D_L(fake_l_pool))
                loss_D_C = lsgan_loss_d(D_C(real_c), D_C(fake_c_pool))
                loss_D = loss_D_L + loss_D_C

            loss_D.backward()
            opt_D.step()

            # Weight clipping for MPS (replaces spectral norm + R1)
            if use_weight_clip:
                D_L.clip_weights(0.01)
                D_C.clip_weights(0.01)

            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("loss/G_total", loss_G_total.item(), global_step)
                writer.add_scalar("loss/G_adv", (loss_adv_G + loss_adv_F).item(), global_step)
                writer.add_scalar("loss/cycle", loss_cyc.item(), global_step)
                writer.add_scalar("loss/identity", loss_id.item(), global_step)
                writer.add_scalar("loss/D", loss_D.item(), global_step)
                writer.add_scalar("lr", opt_G.param_groups[0]["lr"], global_step)
                writer.add_scalar("lr_D", opt_D.param_groups[0]["lr"], global_step)

            if global_step % 500 == 0:
                writer.add_images("mel/clean_input", real_c[:4], global_step)
                writer.add_images("mel/fake_lofi", fake_l[:4], global_step)
                writer.add_images("mel/reconstructed_clean", rec_c[:4], global_step)

        sched_G.step()
        sched_D.step()

        val_cyc = validate(G, F_net, val_clean_loader, val_lofi_loader, device, amp_dtype, use_bf16)
        writer.add_scalar("val/cycle_loss", val_cyc, epoch)

        print(f"Epoch {epoch}/{tc['total_epochs']} | "
              f"G={loss_G_total.item():.4f} cyc={loss_cyc.item():.4f} "
              f"id={loss_id.item():.4f} D={loss_D.item():.4f} "
              f"val_cyc={val_cyc:.4f} lr={opt_G.param_groups[0]['lr']:.6f}")

        save_dict = {
            "epoch": epoch,
            "G": G.state_dict(),
            "F": F_net.state_dict(),
            "D_L": D_L.state_dict(),
            "D_C": D_C.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "ema_G": ema_G.state_dict(),
            "ema_F": ema_F.state_dict(),
            "sched_G": sched_G.state_dict(),
            "sched_D": sched_D.state_dict(),
        }

        if (epoch + 1) % 2 == 0:
            save_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(save_dict, save_path)
            print(f"  Saved {save_path}")
            # Keep only last 2 periodic checkpoints to avoid filling disk
            old_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
            while len(old_ckpts) > 2:
                old_ckpts[0].unlink()
                print(f"  Deleted old checkpoint: {old_ckpts[0].name}")
                old_ckpts.pop(0)

        if val_cyc < best_val_loss:
            best_val_loss = val_cyc
            torch.save(save_dict, ckpt_dir / "best.pt")
            print(f"  New best val cycle loss: {val_cyc:.4f}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
