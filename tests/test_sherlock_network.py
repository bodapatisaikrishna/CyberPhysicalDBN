"""Hand-built classic-pcap tests for src/perception/sherlock_network.py (expected values counted by hand)."""

import struct

import numpy as np
import pytest

from src.perception.sherlock_network import FEATURE_NAMES, pcap_to_bins

F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _eth(dst, src, etype, payload):
    return bytes(dst) + bytes(src) + struct.pack(">H", etype) + payload


def _ipv4(src, dst, proto, l4):
    hdr = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), 0, 0, 64, proto, 0, bytes(src), bytes(dst))
    return hdr + l4


def _tcp(sport, dport, flags, payload=b""):
    return struct.pack(">HHIIBBHHH", sport, dport, 0, 0, 5 << 4, flags, 0, 0, 0) + payload


def _arp(op, smac, sip, tip):
    return struct.pack(">HHBBH", 1, 0x0800, 6, 4, op) + bytes(smac) + bytes(sip) + bytes(6) + bytes(tip)


def _pcap(path, pkts):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, data in pkts:
            s = int(t); us = int(round((t - s) * 1e6))
            f.write(struct.pack("<IIII", s, us, len(data), len(data)) + data)


MAC_A, MAC_B, BC = [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0xFF] * 6


def test_counts_hand_computed(tmp_path):
    t0 = 1000.0
    iec_i = bytes([0x68, 0x04, 0x00, 0x00, 0x00, 0x00])   # control octet 0x00 -> I-frame
    iec_s = bytes([0x68, 0x04, 0x01, 0x00, 0x00, 0x00])   # 0x01 -> S-frame
    iec_u = bytes([0x68, 0x04, 0x07, 0x00, 0x00, 0x00])   # 0x07 -> U-frame
    ipa, ipb = [10, 0, 0, 1], [10, 0, 0, 2]
    pk = [
        (1000.5, _eth(MAC_B, MAC_A, 0x0800, _ipv4(ipa, ipb, 6, _tcp(5000, 2404, 0x18, iec_i)))),
        (1001.0, _eth(MAC_B, MAC_A, 0x0800, _ipv4(ipa, ipb, 6, _tcp(5000, 2404, 0x18, iec_s)))),
        (1001.5, _eth(MAC_A, MAC_B, 0x0800, _ipv4(ipb, ipa, 6, _tcp(2404, 5000, 0x02, b"")))),   # SYN, no payload
        (1002.5, _eth(MAC_B, MAC_A, 0x0800, _ipv4(ipa, ipb, 6, _tcp(5000, 2404, 0x18, iec_u)))),  # bin 1
        (1002.6, _eth(BC, MAC_A, 0x0806, _arp(1, MAC_A, ipa, ipb))),                              # ARP request, bin 1
        (1003.0, _eth(BC, MAC_B, 0x0806, _arp(2, MAC_B, ipa, ipb))),                              # same IP, other MAC
        (1999.0, _eth(BC, MAC_B, 0x0806, _arp(2, MAC_B, ipa, ipb))),                              # outside window
    ]
    p = tmp_path / "x.pcap"; _pcap(p, pk)
    r = pcap_to_bins(p, t0, n_bins=2, bin_s=2.0)
    assert r.n_packets == 7 and r.n_outside_window == 1
    b0, b1 = r.counts[0], r.counts[1]
    assert b0[F["n_pkts"]] == 3 and b1[F["n_pkts"]] == 3
    assert b0[F["n_tcp"]] == 3 and b0[F["n_syn"]] == 1
    assert (b0[F["n_iec104_i"]], b0[F["n_iec104_s"]], b0[F["n_iec104_u"]]) == (1, 1, 0)
    assert b0[F["n_iec104_pkts"]] == 2 and b0[F["iec104_payload_bytes"]] == 12
    assert b1[F["n_iec104_u"]] == 1 and b1[F["n_tcp"]] == 1
    assert b1[F["n_arp"]] == 2 and b1[F["n_arp_request"]] == 1 and b1[F["n_arp_reply"]] == 1
    assert b1[F["n_eth_broadcast"]] == 2
    assert b1[F["arp_ip_multi_mac"]] == 1          # same sender IP, two MACs in one bin
    assert b0[F["arp_ip_multi_mac"]] == 0
    assert b0[F["n_distinct_ip_pairs"]] == 2 and b0[F["n_distinct_src_ip"]] == 2
    assert r.arp_pairs.shape == (2, 3)


def test_rejects_unsupported_pcap(tmp_path):
    p = tmp_path / "bad.pcap"
    p.write_bytes(struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, 1))  # nanosecond magic
    with pytest.raises(ValueError, match="unsupported pcap"):
        pcap_to_bins(p, 0.0, 1)


def test_locf_alignment_hand_computed():
    from src.perception.sherlock_grid import grid_end_times, locf_index, max_locf_staleness, n_before_first

    snaps = np.array([10.0, 12.0, 15.0])
    t_end = grid_end_times(10.0, 4, 2.0)  # 12, 14, 16, 18
    assert t_end.tolist() == [12.0, 14.0, 16.0, 18.0]
    assert locf_index(snaps, t_end).tolist() == [1, 1, 2, 2]  # 12 -> snapshot 12 (<=); 14 -> 12; 16 -> 15; 18 -> 15
    assert n_before_first(snaps, np.array([9.0, 10.0])) == 1
    assert max_locf_staleness(snaps, t_end) == pytest.approx(3.0)


def _asdu13(ca, objs):
    """M_ME_NC_1 ASDU: type 13, VSQ = n objects (SQ=0), COT 3, common address, then IOA(3)+float(4)+QDS(1) each."""
    body = struct.pack("<BBHH", 13, len(objs), 3, ca)
    for ioa, val in objs:
        body += struct.pack("<I", ioa)[:3] + struct.pack("<f", val) + b"\x00"
    apci = bytes([0x68, 4 + len(body), 0x00, 0x00, 0x00, 0x00])  # I-frame
    return apci + body


def test_iec104_measurement_unchanged_counts(tmp_path):
    ipa, ipb = [10, 0, 0, 1], [10, 0, 0, 2]

    def pkt(t, pay):
        return (t, _eth(MAC_B, MAC_A, 0x0800, _ipv4(ipa, ipb, 6, _tcp(2404, 5000, 0x18, pay))))

    p = tmp_path / "m.pcap"
    _pcap(p, [
        pkt(0.1, _asdu13(1, [(100, 1.5), (101, 2.0)])),  # first reports: 2 measurements, 0 unchanged
        pkt(0.5, _asdu13(1, [(100, 1.5)])),              # IOA 100 same value -> unchanged
        pkt(2.1, _asdu13(1, [(101, 2.5), (100, 1.5)])),  # 101 changed; 100 unchanged again
        pkt(2.2, _asdu13(2, [(100, 9.0)])),              # other common address: first report
    ])
    r = pcap_to_bins(p, 0.0, n_bins=2, bin_s=2.0)
    b0, b1 = r.counts
    assert b0[F["n_iec104_measurements"]] == 3 and b0[F["n_iec104_unchanged_measurements"]] == 1
    assert b1[F["n_iec104_measurements"]] == 3 and b1[F["n_iec104_unchanged_measurements"]] == 1
