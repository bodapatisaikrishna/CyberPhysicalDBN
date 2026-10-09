"""Parse the switch pcaps of one Sherlock split (default 02-Semiurban) into 2 s network-feature bins on
the same grid as the physical snapshots (t0 = first physical timestamp). Cached under
data/sherlock/_feature_cache (gitignored). Logs per-file packet counts and timing.

Run: .venv/bin/python -u scripts/build_sherlock_network_features.py [--splits train test]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.perception.sherlock_network import FEATURE_NAMES, pcap_to_bins

CACHE = REPO_ROOT / "data/sherlock/_feature_cache"
BIN_S = 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="02-Semiurban")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max-files", type=int, default=None, help="smoke: only the first N pcaps")
    args = ap.parse_args()
    root = REPO_ROOT / "data/sherlock" / args.scenario / args.scenario
    for split in args.splits:
        phys = np.load(CACHE / f"{args.scenario}__physical_component__{split}__physical.zip.npz", allow_pickle=True)
        pts = phys["timestamps"]
        t0 = float(pts[0])
        n_bins = int((pts[-1] - t0) // BIN_S) + 1
        files = sorted((root / "raw" / split / "pcap").glob("*.pcap"))[: args.max_files]
        total = np.zeros((n_bins, len(FEATURE_NAMES)), dtype=np.float64)
        arps, meta = [], []
        print(f"[{split}] grid t0={t0:.3f}, {n_bins} bins of {BIN_S}s; {len(files)} pcaps", flush=True)
        for f in files:
            t = time.time()
            r = pcap_to_bins(f, t0, n_bins, BIN_S)
            total += r.counts
            arps.append(r.arp_pairs)
            meta.append((f.name, r.n_packets, r.t_first - t0, r.t_last - t0, r.n_outside_window, r.n_truncated))
            print(f"  {f.name}: {r.n_packets} pkts, span [{r.t_first - t0:.1f}, {r.t_last - t0:.1f}]s from grid start, "
                  f"{r.n_outside_window} outside grid, {r.n_truncated} truncated; {time.time() - t:.0f}s", flush=True)
        out = CACHE / f"{args.scenario}__network__{split}.npz"
        np.savez_compressed(
            out, counts=total.astype(np.float32), t0=t0, bin_s=BIN_S, names=np.array(FEATURE_NAMES),
            arp_pairs=np.concatenate(arps) if arps else np.zeros((0, 3), dtype=np.uint64),
            files=np.array([m[0] for m in meta]), n_packets=np.array([m[1] for m in meta]),
        )
        print(f"[{split}] wrote {out.name}; total packets {sum(m[1] for m in meta)}; ARP packets {int(total[:, FEATURE_NAMES.index('n_arp')].sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
