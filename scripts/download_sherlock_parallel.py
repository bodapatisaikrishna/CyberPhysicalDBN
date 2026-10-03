#!/usr/bin/env python3
"""Resumable, parallel-range downloader for the Sherlock dataset (Zenodo record
15168928, v1), with the same md5 gate as scripts/download_sherlock.sh.

WHY A SECOND DOWNLOADER. Zenodo throttles each connection to roughly
0.25-0.3 MB/s (measured 2026-09-28) and pushes back on very many parallel
streams; scripts/download_sherlock.sh's single `curl` needs ~4 h uninterrupted
for 02-Semiurban (4.7 GB) and cannot resume after a dropped connection. This
script fetches fixed-size chunks on N threads and writes them in place at their
offsets; finished chunk ids are appended to `<file>.done`, so an interrupted run
resumes instead of restarting. Stdlib only -- no new dependency.

TWO MODES.
  full (default)  the whole zip, md5-gated against the hash Zenodo publishes,
                  then CRC-tested and extracted; the zip is deleted afterwards.
  --essential     only the members this project READS -- every `*.state.gz`
                  plus every member <= 20 MB compressed (event catalogs,
                  data-point maps, configs, docs) -- fetched by HTTP range
                  straight out of the remote zip, each verified by its zip
                  CRC-32 and size. Skips the pcaps and the nested
                  `physical.zip` / `control-center.zip` (~5 GB of 02-Semiurban's
                  4.7 GB zip is never read by any experiment here). Integrity is
                  per member (CRC-32), not the whole-file md5, which needs the
                  whole zip; run the full mode later if the raw captures matter.

DISK (full mode). A scenario needs its zip PLUS its extracted tree at once
(measured: 01-Basic 0.70 GB -> 3.2 GB; 02-Semiurban 4.68 GB -> 6.05 GB;
03-Rural 1.87 GB -> 2.51 GB). A pre-flight check refuses to start below
zip + extracted + 1 GiB free.

Like download_sherlock.sh this fetches real bytes from the internet: an
explicit-permission action, never invoked by an experiment or test.

Usage:
  scripts/download_sherlock_parallel.py [--scenario 01-Basic|02-Semiurban|03-Rural|all]
                                        [--essential] [--paper] [--out data/sherlock]
                                        [--workers 16] [--chunk-mib 32] [--keep-zip]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/records/15168928/files"
ESSENTIAL_MAX_COMPRESSED = 20_000_000  # bytes; state files are always essential

# name -> (md5 published on Zenodo, zip bytes, extracted bytes as measured)
SCENARIOS = {
    "01-Basic": ("4f751246a245b952f0200e74ef1da10f", 704_090_998, 3_200_000_000),
    "02-Semiurban": ("e864944c52fb4a6f27b544c08a351ae7", 4_677_487_711, 6_050_000_000),
    "03-Rural": ("2925a5275ef63d9a413a218fc667fd44", 1_866_012_067, 2_510_000_000),
}
PAPER = ("paper.pdf", "662db881140984b51952d674daac4a25")


def with_retries(fn, what: str):
    """Run `fn()` until it succeeds: any network error is retried with capped
    exponential backoff (<= 60 s), 240 attempts (~4 h) before giving up. Every
    request in this script goes through here or through the chunk loop in
    `fetch_range`, so a dropped connection never aborts a resumable download."""
    for attempt in range(240):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            wait = min(60, 2 ** attempt)
            print(f"  {what}: attempt {attempt + 1} failed ({e!r}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after 240 attempts")


def head_size(url: str) -> int:
    def go() -> int:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            return int(r.headers["Content-Length"])
    return with_retries(go, "HEAD")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch_range(url: str, dest: Path, first: int, last: int, workers: int, chunk_bytes: int, label: str) -> None:
    """Parallel, resumable download of bytes `[first, last]` (inclusive) of
    `url` into `dest`, stored at offset 0 of `dest` (i.e. `dest` holds exactly
    that slice). Progress is tracked per chunk in `<dest>.done`."""
    size = last - first + 1
    n_chunks = (size + chunk_bytes - 1) // chunk_bytes
    done_path = Path(str(dest) + ".done")
    done: set[int] = set()
    if dest.exists() and done_path.exists() and dest.stat().st_size == size:
        done = {int(x) for x in done_path.read_text().split()}
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.truncate(size)
        done_path.write_text("")
    print(f"  {label}: {size / 1e6:.1f} MB in {n_chunks} chunks, {len(done)} already done, {workers} workers", flush=True)

    fd = os.open(dest, os.O_RDWR)
    lock = threading.Lock()
    got = [sum(min(chunk_bytes, size - i * chunk_bytes) for i in done)]

    def fetch(i: int) -> None:
        if i in done:
            return
        lo = i * chunk_bytes
        hi = min(lo + chunk_bytes, size) - 1
        # A resumable downloader should wait out an outage (laptop sleep, Wi-Fi
        # drop) rather than die: 240 attempts at <=60 s backoff is ~4 h of patience.
        for attempt in range(240):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={first + lo}-{first + hi}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    pos = lo
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        os.pwrite(fd, block, pos)
                        pos += len(block)
                if pos != hi + 1:
                    raise IOError(f"short read on chunk {i}: {pos - lo} of {hi + 1 - lo} bytes")
                with lock:
                    got[0] += hi + 1 - lo
                    with open(done_path, "a") as d:
                        d.write(f"{i}\n")
                return
            except Exception as e:  # noqa: BLE001 -- retry any network error, then fail loudly
                wait = min(60, 2 ** attempt)
                print(f"  chunk {i} attempt {attempt + 1} failed ({e!r}); retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"chunk {i} failed after 240 attempts")

    stop = threading.Event()

    def progress() -> None:
        t0, b0 = time.time(), got[0]
        while not stop.wait(60):
            rate = (got[0] - b0) / max(time.time() - t0, 1e-9)
            print(f"  {label}: {got[0] / 1e6:8.1f}/{size / 1e6:.1f} MB  {rate / 1e6:5.2f} MB/s  "
                  f"eta {(size - got[0]) / max(rate, 1) / 60:5.1f} min (finished chunks only)", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, range(n_chunks)))
    stop.set()
    os.close(fd)
    done_path.unlink(missing_ok=True)


def fetch_verified(url: str, dest: Path, expected_md5: str, workers: int, chunk_bytes: int) -> None:
    """Whole-file parallel download, md5-gated. Raises SystemExit(1) on mismatch."""
    size = head_size(url)
    fetch_range(url, dest, 0, size - 1, workers, chunk_bytes, dest.name)
    actual = md5_of(dest)
    print(f"  md5 actual={actual} expected={expected_md5}", flush=True)
    if actual != expected_md5:
        print(f"MD5 MISMATCH for {dest.name} -- file NOT trusted, nothing extracted.", file=sys.stderr)
        raise SystemExit(1)


class HttpRangeFile:
    """Seekable read-only file over HTTP range requests with a small block
    cache -- enough for `zipfile` to read a remote zip's central directory."""

    BLOCK = 4 << 20

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = head_size(url)
        self._cache: tuple[int | None, bytes] = (None, b"")

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        def go() -> bytes:
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        return with_retries(go, f"central-directory range {start}-{end}")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        out = b""
        while self.pos < end:
            cstart, cdata = self._cache
            if cstart is None or not (cstart <= self.pos < cstart + len(cdata)):
                bstart = (self.pos // self.BLOCK) * self.BLOCK
                cdata = self._fetch(bstart, min(bstart + self.BLOCK, self.size) - 1)
                self._cache, cstart = (bstart, cdata), bstart
            take = min(end, cstart + len(cdata)) - self.pos
            out += cdata[self.pos - cstart: self.pos - cstart + take]
            self.pos += take
        return out


def _crc_ok(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.exists() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            crc = zlib.crc32(block, crc)
    return (crc & 0xFFFFFFFF) == info.CRC


def extract_member_from_remote(url: str, zf: zipfile.ZipFile, info: zipfile.ZipInfo, out_root: Path,
                               workers: int, chunk_bytes: int, tmp_dir: Path) -> None:
    """Range-download one member's compressed bytes, inflate it to its final
    path and verify the zip's own CRC-32 and size. A failed check deletes the
    output and raises -- a member that does not verify is never left in place."""
    dest = (out_root / info.filename).resolve()
    if out_root.resolve() not in dest.parents:
        raise SystemExit(f"refusing to extract outside {out_root}: {info.filename!r}")
    if _crc_ok(dest, info):
        print(f"  {info.filename}: already present and CRC-verified", flush=True)
        return
    # local file header: 30 fixed bytes, then filename + extra field, then the data
    def read_header() -> bytes:  # exactly the 30 fixed header bytes, not a 4 MB cache block
        req = urllib.request.Request(url, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + 29}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    fixed = with_retries(read_header, f"local header of {info.filename}")
    if fixed[:4] != b"PK\x03\x04":
        raise SystemExit(f"bad local header for {info.filename}")
    fn_len, extra_len = int.from_bytes(fixed[26:28], "little"), int.from_bytes(fixed[28:30], "little")
    data_start = info.header_offset + 30 + fn_len + extra_len
    part = tmp_dir / (info.filename.replace("/", "__") + ".part")
    if info.compress_size > 0:
        fetch_range(url, part, data_start, data_start + info.compress_size - 1, workers, chunk_bytes, info.filename)
    else:
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"")
    dest.parent.mkdir(parents=True, exist_ok=True)
    crc, n = 0, 0
    with open(part, "rb") as src, open(dest, "wb") as out:
        if info.compress_type == zipfile.ZIP_DEFLATED:
            d = zlib.decompressobj(-15)
            for block in iter(lambda: src.read(1 << 22), b""):
                raw = d.decompress(block)
                out.write(raw); crc = zlib.crc32(raw, crc); n += len(raw)
            raw = d.flush()
            out.write(raw); crc = zlib.crc32(raw, crc); n += len(raw)
        elif info.compress_type == zipfile.ZIP_STORED:
            for block in iter(lambda: src.read(1 << 22), b""):
                out.write(block); crc = zlib.crc32(block, crc); n += len(block)
        else:
            raise SystemExit(f"unsupported zip method {info.compress_type} for {info.filename}")
    part.unlink()
    if (crc & 0xFFFFFFFF) != info.CRC or n != info.file_size:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"CRC/size MISMATCH for {info.filename}: got crc={crc & 0xFFFFFFFF:08x} size={n}, "
                         f"zip says crc={info.CRC:08x} size={info.file_size}")
    print(f"  {info.filename}: {n / 1e6:.1f} MB, CRC-32 verified", flush=True)


def fetch_essential(name: str, out: Path, workers: int, chunk_bytes: int, all_members: bool = False) -> None:
    url = f"{ZENODO_BASE}/{name}.zip"
    print(f"==> {name} (essential members only)", flush=True)
    zf = zipfile.ZipFile(HttpRangeFile(url))
    members = [i for i in zf.infolist()
               if not i.is_dir() and (all_members or i.filename.endswith(".state.gz") or i.compress_size <= ESSENTIAL_MAX_COMPRESSED)]
    skipped = [i for i in zf.infolist() if not i.is_dir() and i not in members]
    need = sum(i.file_size for i in members) if all_members else sum(i.compress_size for i in members)
    free = shutil.disk_usage(out if out.exists() else out.parent).free
    print(f"  {len(members)} essential members, {need / 1e6:.0f} MB compressed; skipping {len(skipped)} large members "
          f"({sum(i.compress_size for i in skipped) / 1e9:.2f} GB); {free / 2**30:.1f} GiB free", flush=True)
    want = need + (2 << 30) if all_members else 3 * need
    # members already on disk and CRC-verified need no space; only still-missing ones count
    if all_members:
        want = sum(i.file_size for i in members if not _crc_ok((out / name / i.filename), i)) + (1 << 30)
    if free < want:
        raise SystemExit(f"not enough free disk for {name}: {free / 2**30:.1f} GiB free, want >= {want / 2**30:.1f} GiB")
    tmp = out / f".{name}.parts"
    # big members first (the state files), each with its own parallel fetch
    for info in sorted(members, key=lambda i: -i.compress_size):
        # `out/<name>/<member path>` -- the same doubly-nested layout as the full mode
        # (`extractall(out/<name>)`) and 01-Basic, so configs/sherlock_full.yaml's
        # `data/sherlock/<name>/<name>/` is right in both modes.
        extract_member_from_remote(url, zf, info, out / name, workers, chunk_bytes, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"==> {name} essential members ready under {out / name}/ "
          f"(skipped: {[i.filename for i in skipped][:3]} ... {len(skipped)} total)", flush=True)


def preflight(name: str, out: Path) -> None:
    _, zip_b, ext_b = SCENARIOS[name]
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    need = zip_b + ext_b + (1 << 30)
    print(f"  disk: {free / 2**30:.1f} GiB free, need ~{need / 2**30:.1f} GiB (zip + extracted + 1 GiB margin)")
    if free < need:
        raise SystemExit(f"not enough free disk for {name}: {free / 2**30:.1f} GiB < {need / 2**30:.1f} GiB")


def fetch_scenario(name: str, out: Path, workers: int, chunk_bytes: int, keep_zip: bool) -> None:
    md5, _, _ = SCENARIOS[name]
    print(f"==> {name}", flush=True)
    preflight(name, out)
    zip_path = out / f"{name}.zip"
    fetch_verified(f"{ZENODO_BASE}/{name}.zip", zip_path, md5, workers, chunk_bytes)
    print(f"  extracting into {out / name}/ ...", flush=True)
    (out / name).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()  # per-member CRC32 check on top of the whole-file md5
        if bad is not None:
            raise SystemExit(f"CRC failure on member {bad!r} in {zip_path.name}")
        zf.extractall(out / name)
    if not keep_zip:
        zip_path.unlink()
    print(f"==> {name} ready at {out / name}/", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", choices=[*SCENARIOS, "all"], default=None)
    ap.add_argument("--essential", action="store_true", help="only the members the experiments read (see module docstring)")
    ap.add_argument("--all-members", action="store_true", help="with --essential: EVERY member, still one at a time, CRC-verified, no zip kept "
                    "(nested physical.zip/control-center.zip are stored as the zips they are, not unpacked)")
    ap.add_argument("--paper", action="store_true", help="also fetch paper.pdf (1.7 MB)")
    ap.add_argument("--out", default="data/sherlock")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mib", type=int, default=32)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    if args.scenario is None and not args.paper:
        ap.error("nothing to do: pass --scenario and/or --paper")

    out, chunk = Path(args.out), args.chunk_mib << 20
    for name in (SCENARIOS if args.scenario == "all" else [args.scenario] if args.scenario else []):
        if args.essential or args.all_members:
            out.mkdir(parents=True, exist_ok=True)
            fetch_essential(name, out, args.workers, chunk, args.all_members)
        else:
            fetch_scenario(name, out, args.workers, chunk, args.keep_zip)
    if args.paper:
        print("==> paper.pdf", flush=True)
        fetch_verified(f"{ZENODO_BASE}/{PAPER[0]}", out / PAPER[0], PAPER[1], min(args.workers, 4), 1 << 20)
    print("Done. See docs/sherlock_download.md for dataset notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
