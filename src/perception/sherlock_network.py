"""Network-capture view of a Sherlock scenario: classic-pcap mirror captures ->
uniform-time-bin traffic features (LAB_NOTEBOOK.md 2026-10-05, exp17).

Why: the physical-state views (exp13-16) show no footprint for arp-spoof,
control-and-freeze and drift-off. Those act on the network / control path, so the
packet captures (`raw/<split>/pcap/switch-*.pcap`, six switch mirror ports per split,
Ethernet, classic libpcap) are the remaining place to look.

No dependency: a classic-pcap record walk plus numpy vectorised header extraction
(Ethernet / IPv4 / TCP / ARP). Per-bin features are counts, so the view is natively
uniform in time (no interpolation) -- unlike the ~2 s jittered physical snapshots.

Feature definitions are fixed here, before any label is looked at. IEC 60870-5-104
frames are recognised by TCP port 2404 with start byte 0x68; frame type from the first
control octet: I-frame (bit0 == 0), S-frame (low bits 01), U-frame (low bits 11).
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PCAP_MAGIC_LE = 0xA1B2C3D4  # microsecond timestamps, little-endian (as in the shipped files)
IEC104_PORT = 2404

FEATURE_NAMES: tuple[str, ...] = (
    "n_pkts", "n_bytes", "n_ipv4", "n_ipv6", "n_arp", "n_arp_request", "n_arp_reply", "n_other_eth",
    "n_tcp", "n_udp", "n_icmp", "n_syn", "n_rst", "n_fin",
    "n_iec104_pkts", "n_iec104_i", "n_iec104_s", "n_iec104_u", "iec104_payload_bytes",
    "n_eth_broadcast", "n_distinct_ip_pairs", "n_distinct_src_ip", "arp_ip_multi_mac",
)


@dataclass(frozen=True)
class PcapBins:
    counts: np.ndarray  # [n_bins, len(FEATURE_NAMES)] float32
    arp_pairs: np.ndarray  # [m, 3] uint64: (bin, sender_ip_u32, sender_mac_u48) for every ARP packet
    n_packets: int
    t_first: float
    t_last: float
    n_outside_window: int  # packets whose timestamp falls outside [t0, t0 + n_bins*bin_s)
    n_truncated: int  # packets too short for the headers a feature needs (kept in n_pkts/n_bytes)


def read_records(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.memmap]:
    """Walk the classic-pcap record headers. Returns `(ts[float64], data_offset[int64],
    caplen[int64], wirelen[int64], buf)` with `buf` the file as uint8. Raises on any
    format other than little-endian microsecond Ethernet pcap -- never guesses."""
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    magic, vmaj, vmin, _tz, _sig, _snap, link = struct.unpack_from("<IHHiIII", buf, 0)
    if magic != PCAP_MAGIC_LE or link != 1:
        raise ValueError(f"{path}: unsupported pcap (magic {magic:#x}, linktype {link}); expected LE micro, Ethernet")
    n = buf.shape[0]
    mv = memoryview(buf)
    ts, off, cap, wire = [], [], [], []
    pos = 24
    unpack = struct.Struct("<IIII").unpack_from
    while pos + 16 <= n:
        s, us, c, w = unpack(mv, pos)
        if pos + 16 + c > n:
            break  # truncated final record
        ts.append(s + us * 1e-6)
        off.append(pos + 16)
        cap.append(c)
        wire.append(w)
        pos += 16 + c
    return (np.asarray(ts, dtype=np.float64), np.asarray(off, dtype=np.int64),
            np.asarray(cap, dtype=np.int64), np.asarray(wire, dtype=np.int64), buf)


def _u8(buf, idx):  # gather bytes (idx already bounds-safe)
    return buf[idx].astype(np.int64)


def pcap_to_bins(path: Path, t0: float, n_bins: int, bin_s: float = 2.0, chunk: int = 1_000_000) -> PcapBins:
    """Count features per `bin_s`-second bin on the grid `[t0 + k*bin_s, t0 + (k+1)*bin_s)`."""
    ts, off, cap, wire, buf = read_records(path)
    F = {name: i for i, name in enumerate(FEATURE_NAMES)}
    counts = np.zeros((n_bins, len(FEATURE_NAMES)), dtype=np.float64)
    pair_keys: list[np.ndarray] = []  # (bin, src_u32, dst_u32) per IPv4 packet, uniqued per chunk later
    src_keys: list[np.ndarray] = []
    arp_rows: list[np.ndarray] = []
    n_out = n_trunc = 0
    bins_all = np.floor((ts - t0) / bin_s).astype(np.int64)
    for a in range(0, len(ts), chunk):
        sl = slice(a, a + chunk)
        b = bins_all[sl]
        ok = (b >= 0) & (b < n_bins)
        n_out += int((~ok).sum())
        o, c, w, b = off[sl][ok], cap[sl][ok], wire[sl][ok], b[ok]
        if o.size == 0:
            continue
        np.add.at(counts[:, F["n_pkts"]], b, 1.0)
        np.add.at(counts[:, F["n_bytes"]], b, w.astype(np.float64))
        short = c < 14
        n_trunc += int(short.sum())
        og = np.where(short, 0, o)  # safe dummy offset for gathers, masked below
        etype = (_u8(buf, og + 12) << 8) | _u8(buf, og + 13)
        etype[short] = -1
        bc = np.ones(o.size, dtype=bool)
        for k in range(6):
            bc &= _u8(buf, og + k) == 0xFF
        bc &= ~short
        np.add.at(counts[:, F["n_eth_broadcast"]], b[bc], 1.0)
        ip4, ip6, arp = etype == 0x0800, etype == 0x86DD, etype == 0x0806
        np.add.at(counts[:, F["n_ipv4"]], b[ip4], 1.0)
        np.add.at(counts[:, F["n_ipv6"]], b[ip6], 1.0)
        np.add.at(counts[:, F["n_arp"]], b[arp], 1.0)
        np.add.at(counts[:, F["n_other_eth"]], b[~(ip4 | ip6 | arp) & ~short], 1.0)

        # ARP: need 42 captured bytes
        am = arp & (c >= 42)
        if am.any():
            oa, ba = o[am], b[am]
            op = (_u8(buf, oa + 20) << 8) | _u8(buf, oa + 21)
            np.add.at(counts[:, F["n_arp_request"]], ba[op == 1], 1.0)
            np.add.at(counts[:, F["n_arp_reply"]], ba[op == 2], 1.0)
            ip_s = np.zeros(oa.size, dtype=np.uint64)
            mac_s = np.zeros(oa.size, dtype=np.uint64)
            for k in range(4):
                ip_s = (ip_s << np.uint64(8)) | _u8(buf, oa + 28 + k).astype(np.uint64)
            for k in range(6):
                mac_s = (mac_s << np.uint64(8)) | _u8(buf, oa + 22 + k).astype(np.uint64)
            arp_rows.append(np.stack([ba.astype(np.uint64), ip_s, mac_s], axis=1))

        # IPv4 (+ TCP/UDP/ICMP): need 34 bytes for the IP header
        im = ip4 & (c >= 34)
        if im.any():
            oi, bi, ci = o[im], b[im], c[im]
            ihl = (_u8(buf, oi + 14) & 0x0F) * 4
            proto = _u8(buf, oi + 23)
            tot = (_u8(buf, oi + 16) << 8) | _u8(buf, oi + 17)
            src = np.zeros(oi.size, dtype=np.uint64)
            dst = np.zeros(oi.size, dtype=np.uint64)
            for k in range(4):
                src = (src << np.uint64(8)) | _u8(buf, oi + 26 + k).astype(np.uint64)
                dst = (dst << np.uint64(8)) | _u8(buf, oi + 30 + k).astype(np.uint64)
            pair_keys.append(np.stack([bi.astype(np.uint64), src, dst], axis=1))
            src_keys.append(np.stack([bi.astype(np.uint64), src], axis=1))
            np.add.at(counts[:, F["n_tcp"]], bi[proto == 6], 1.0)
            np.add.at(counts[:, F["n_udp"]], bi[proto == 17], 1.0)
            np.add.at(counts[:, F["n_icmp"]], bi[proto == 1], 1.0)
            tm = (proto == 6) & (ci >= 14 + ihl + 20)
            n_trunc += int(((proto == 6) & ~tm).sum())
            if tm.any():
                ot, bt, iht, tott = oi[tm], bi[tm], ihl[tm], tot[tm]
                tb = ot + 14 + iht
                sport = (_u8(buf, tb) << 8) | _u8(buf, tb + 1)
                dport = (_u8(buf, tb + 2) << 8) | _u8(buf, tb + 3)
                flags = _u8(buf, tb + 13)
                thl = (_u8(buf, tb + 12) >> 4) * 4
                pay = tott - iht - thl
                np.add.at(counts[:, F["n_syn"]], bt[(flags & 0x02) > 0], 1.0)
                np.add.at(counts[:, F["n_rst"]], bt[(flags & 0x04) > 0], 1.0)
                np.add.at(counts[:, F["n_fin"]], bt[(flags & 0x01) > 0], 1.0)
                is104 = ((sport == IEC104_PORT) | (dport == IEC104_PORT)) & (pay >= 6)
                avail = (c[im][tm] >= 14 + iht + thl + 6)
                is104 &= avail
                if is104.any():
                    pb = tb + thl
                    start_ok = _u8(buf, np.where(is104, pb, 0)) == 0x68
                    m104 = is104 & start_ok
                    ctrl = _u8(buf, np.where(m104, pb + 2, 0))
                    bb = bt[m104]
                    cc = ctrl[m104]
                    np.add.at(counts[:, F["n_iec104_pkts"]], bb, 1.0)
                    np.add.at(counts[:, F["n_iec104_i"]], bb[(cc & 1) == 0], 1.0)
                    np.add.at(counts[:, F["n_iec104_s"]], bb[(cc & 3) == 1], 1.0)
                    np.add.at(counts[:, F["n_iec104_u"]], bb[(cc & 3) == 3], 1.0)
                    np.add.at(counts[:, F["iec104_payload_bytes"]], bb, pay[m104].astype(np.float64))
    # distinct IP pairs / sources / multi-MAC ARP per bin
    if pair_keys:
        pk = np.unique(np.concatenate(pair_keys), axis=0)
        np.add.at(counts[:, F["n_distinct_ip_pairs"]], pk[:, 0].astype(np.int64), 1.0)
        sk = np.unique(np.concatenate(src_keys), axis=0)
        np.add.at(counts[:, F["n_distinct_src_ip"]], sk[:, 0].astype(np.int64), 1.0)
    arp_pairs = np.concatenate(arp_rows) if arp_rows else np.zeros((0, 3), dtype=np.uint64)
    if len(arp_pairs):
        up = np.unique(arp_pairs, axis=0)  # distinct (bin, ip, mac)
        ipb = np.unique(up[:, :2], axis=0, return_counts=False)
        # count IPs with >1 distinct MAC in a bin
        key = up[:, 0] * np.uint64(1 << 32) + up[:, 1]
        uk, cnt = np.unique(key, return_counts=True)
        multi_bins = (uk[cnt > 1] >> np.uint64(32)).astype(np.int64)
        np.add.at(counts[:, F["arp_ip_multi_mac"]], multi_bins, 1.0)
    return PcapBins(
        counts=counts.astype(np.float32), arp_pairs=arp_pairs, n_packets=int(len(ts)),
        t_first=float(ts[0]) if len(ts) else float("nan"), t_last=float(ts[-1]) if len(ts) else float("nan"),
        n_outside_window=n_out, n_truncated=n_trunc,
    )
