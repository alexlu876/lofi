"""Measure z.std on a random sample of LMDB entries via the pretrained AE.
Informs whether sigma_data=0.65 is appropriate for the lofi corpus."""
import sys, random, json
from pathlib import Path
import numpy as np
import torch
import lmdb

sys.path.insert(0, "/Users/alex/lu/git/lofi/lofi-transfer/ctd_repo/dataset")
from audio_example.audio_example import AudioExample

LMDB_PATH = "/Users/alex/lu/git/lofi/lofi-transfer/data/lmdb/lofi"
AE_PATH = "/Users/alex/lu/git/lofi/lofi-transfer/pretrained/AE_real_instruments.pt"
X_LENGTH = 131072
N_SAMPLES = 100

torch.set_grad_enabled(False)
ae = torch.jit.load(AE_PATH, map_location="cpu").eval()

env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
with env.begin() as txn:
    n_entries = txn.stat()["entries"]
    print(f"LMDB entries: {n_entries}")
    keys = []
    cursor = txn.cursor()
    for k, _ in cursor:
        keys.append(k)
        if len(keys) >= min(n_entries, 5000):
            break
    sample_keys = random.sample(keys, min(N_SAMPLES, len(keys)))

z_stds, z_means, z_mins, z_maxs = [], [], [], []
with env.begin() as txn:
    for k in sample_keys:
        v = txn.get(k)
        ae_entry = AudioExample(v)
        wav = ae_entry.get("waveform")  # shape (1, num_signal)
        x = torch.from_numpy(wav).float()
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 1, T]
        # Sample a random 131072 window from within the larger entry
        total_T = x.shape[-1]
        if total_T > X_LENGTH:
            start = random.randint(0, total_T - X_LENGTH)
            x = x[..., start:start + X_LENGTH]
        elif total_T < X_LENGTH:
            x = torch.nn.functional.pad(x, (0, X_LENGTH - total_T))
        z = ae.encode(x)
        z_stds.append(z.std().item())
        z_means.append(z.mean().item())
        z_mins.append(z.min().item())
        z_maxs.append(z.max().item())

out = {
    "n_entries": int(n_entries),
    "n_sampled": len(z_stds),
    "z_std_mean": float(np.mean(z_stds)),
    "z_std_median": float(np.median(z_stds)),
    "z_std_min": float(np.min(z_stds)),
    "z_std_max": float(np.max(z_stds)),
    "z_mean_mean": float(np.mean(z_means)),
    "z_min": float(np.min(z_mins)),
    "z_max": float(np.max(z_maxs)),
    "sigma_data_current": 0.65,
    "verdict": "OK" if 0.5 <= float(np.mean(z_stds)) <= 0.95 else "RECONSIDER_SIGMA_DATA",
}
print(json.dumps(out, indent=2))
Path("/tmp/lofi_z_std.json").write_text(json.dumps(out, indent=2))
