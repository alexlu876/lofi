"""Phase 2: HiFi-GAN fine-tuning (spec §6, Phase 2)."""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.wav_dataset import WavDataset
from src.losses.hifigan_losses import discriminator_loss, feature_matching_loss, generator_loss
from src.models.hifigan import HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator

SAMPLE_RATE = 22050


def build_mel_transform(cfg, device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        n_mels=cfg["audio"]["n_mels"],
        f_min=cfg["audio"]["f_min"],
        f_max=cfg["audio"]["f_max"],
        norm="slaney",
        mel_scale="slaney",
    ).to(device)


def mel_from_wav(wav, mel_transform, floor=1e-5):
    mel = mel_transform(wav)
    return torch.log(torch.clamp(mel, min=floor))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--wav-dir", required=True, help="Path to WAV files for fine-tuning")
    parser.add_argument("--init-ckpt", default=None, help="UNIVERSAL_V1 checkpoint directory")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-dir", default="logs/tensorboard/hifigan")
    parser.add_argument("--ckpt-dir", default="checkpoints/hifigan")
    parser.add_argument("--max-steps", type=int, default=200000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hc = cfg["hifigan"]["training"]
    hl = cfg["hifigan"]["loss"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = WavDataset(args.wav_dir, segment_size=hc["segment_size"])
    loader = DataLoader(dataset, batch_size=hc["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    generator = HiFiGANGenerator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if args.init_ckpt:
        from src.models.hifigan.utils import load_hifigan_generator
        pretrained = load_hifigan_generator(args.init_ckpt, device)
        generator.load_state_dict(pretrained.state_dict())
        print(f"Loaded UNIVERSAL_V1 from {args.init_ckpt}")

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=hc["lr"],
        betas=tuple(hc["betas"]), weight_decay=hc["weight_decay"],
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=hc["lr"], betas=tuple(hc["betas"]), weight_decay=hc["weight_decay"],
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=hc["lr_decay"])
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=hc["lr_decay"])

    mel_transform = build_mel_transform(cfg, device)
    writer = SummaryWriter(args.log_dir)
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
        if "sched_d" in ckpt:
            sched_d.load_state_dict(ckpt["sched_d"])
        global_step = ckpt["step"]
        print(f"Resumed from step {global_step}")

    lambda_fm = hl["lambda_fm"]
    lambda_mel = hl["lambda_mel"]

    generator.train()
    mpd.train()
    msd.train()

    while global_step < args.max_steps:
        for wav in loader:
            if global_step >= args.max_steps:
                break

            wav = wav.to(device)  # [B, 1, segment_size]
            mel_real = mel_from_wav(wav.squeeze(1), mel_transform)  # [B, 80, T_mel]
            wav_gen = generator(mel_real)  # [B, 1, T_wav]

            min_len = min(wav.shape[-1], wav_gen.shape[-1])
            wav = wav[..., :min_len]
            wav_gen_d = wav_gen[..., :min_len]

            # --- Discriminator step ---
            opt_d.zero_grad()
            mpd_real, mpd_gen, _, _ = mpd(wav, wav_gen_d.detach())
            msd_real, msd_gen, _, _ = msd(wav, wav_gen_d.detach())
            loss_d = discriminator_loss(mpd_real, mpd_gen) + discriminator_loss(msd_real, msd_gen)
            loss_d.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            mpd_real, mpd_gen, mpd_fmap_r, mpd_fmap_g = mpd(wav, wav_gen_d)
            msd_real, msd_gen, msd_fmap_r, msd_fmap_g = msd(wav, wav_gen_d)

            loss_g_adv = generator_loss(mpd_gen) + generator_loss(msd_gen)
            loss_fm = feature_matching_loss(mpd_fmap_r, mpd_fmap_g) + feature_matching_loss(msd_fmap_r, msd_fmap_g)

            mel_gen = mel_from_wav(wav_gen.squeeze(1), mel_transform)
            min_mel_t = min(mel_real.shape[-1], mel_gen.shape[-1])
            loss_mel = torch.nn.functional.l1_loss(mel_gen[..., :min_mel_t], mel_real[..., :min_mel_t])

            loss_g = loss_g_adv + lambda_fm * loss_fm + lambda_mel * loss_mel
            loss_g.backward()
            opt_g.step()

            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("hifigan/loss_g", loss_g.item(), global_step)
                writer.add_scalar("hifigan/loss_d", loss_d.item(), global_step)
                writer.add_scalar("hifigan/loss_mel", loss_mel.item(), global_step)
                writer.add_scalar("hifigan/loss_fm", loss_fm.item(), global_step)
                print(f"Step {global_step}/{args.max_steps} | "
                      f"G={loss_g.item():.4f} D={loss_d.item():.4f} mel={loss_mel.item():.4f}")

            if global_step % 10000 == 0:
                save_path = ckpt_dir / f"step_{global_step:07d}.pt"
                torch.save({
                    "step": global_step,
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "sched_g": sched_g.state_dict(),
                    "sched_d": sched_d.state_dict(),
                }, save_path)
                print(f"  Saved {save_path}")

        sched_g.step()
        sched_d.step()

    writer.close()
    print("HiFi-GAN fine-tuning complete.")


if __name__ == "__main__":
    main()
