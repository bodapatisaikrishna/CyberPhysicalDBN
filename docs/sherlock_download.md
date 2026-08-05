# Sherlock dataset — download notes

Session 7 (LAB_NOTEBOOK.md 2026-08-05). This file documents the manual
download steps and every discrepancy found between the task's own
description of Sherlock and what the live site/Zenodo actually serve,
verified directly rather than assumed (CLAUDE.md rules 2/5).

## What Sherlock is

Wagner, E., Bader, L., Wolsing, K., Serror, M. "Sherlock: A Dataset for
Process-aware Intrusion Detection Research on Power Grid Networks." ACM
CODASPY'25. Built with the Wattson power-grid co-simulator (PowerOwl on
pandapower for steady-state power flow; Docker-namespace hosts/switches/
routers with `tc` link impairment for the network layer; IEC 60870-5-104
for control-center<->RTU/MTU communication). Documentation:
https://sherlock.wattson.it/

## Getting the files

```bash
scripts/download_sherlock.sh                       # 01-Basic only (704.1 MB), the default
scripts/download_sherlock.sh --scenario 02-Semiurban  # 4.7 GB, opt-in
scripts/download_sherlock.sh --scenario 03-Rural      # 1.9 GB, opt-in
scripts/download_sherlock.sh --scenario all --with-paper
```

Running this script downloads real bytes from the internet and is an
explicit-permission action (CLAUDE.md safety rules) — a human, or an agent
that has just been told in chat exactly which file/source/size to fetch,
should run it, never as a side effect of running an experiment.

Files land at `data/sherlock/<scenario>/` (already covered by
`.gitignore`'s `data/*` rule — never commit them). The script verifies each
zip's md5 against the hash published on Zenodo before extracting, and fails
loudly (never silently proceeds) on a mismatch.

## Source and exact files (Zenodo record `15168928`, v1)

The live https://sherlock.wattson.it/download/ page links to
https://zenodo.org/records/15168928. Verified directly:

| file | size | md5 |
|---|---|---|
| `01-Basic.zip` | 704.1 MB (704,090,998 bytes, confirmed via `curl -I`) | `4f751246a245b952f0200e74ef1da10f` |
| `02-Semiurban.zip` | 4.7 GB | `e864944c52fb4a6f27b544c08a351ae7` |
| `03-Rural.zip` | 1.9 GB | `2925a5275ef63d9a413a218fc667fd44` |
| `paper.pdf` | 1.7 MB | `662db881140984b51952d674daac4a25` |

Total 7.2 GB. License: CC-BY 4.0. Download URLs follow
`https://zenodo.org/records/15168928/files/<name>`, confirmed by a direct
`HEAD` request (200 OK, matching `content-length` and `content-disposition`)
before this script was trusted to use that pattern.

## Discrepancies against this project's own task description — reported, not silently resolved

1. **"35 days" vs. the site's own "over 30 days."** The Sherlock task text
   given to this project says the dataset spans 35 days. The
   Testbed-Setup page at sherlock.wattson.it states: "In total, Sherlock
   contains simulation results from over 30 days of power grid behavior,
   split into training and test set for a total of 3 different scenarios."
   That is a total across all 3 scenarios combined, not per-scenario, and
   it does not say 35. Used "over 30 days" (the site's own wording) rather
   than silently adopting the task text's number.

2. **Dataset version.** The download page links to Zenodo record
   `15168928`, which Zenodo itself labels **v1** (published 2025-04-10).
   Zenodo's own UI states a newer version is available: v2
   (`15260901`, 2025-04-22) and v3 (`18467070`, 2026-02-04, the latest as
   of this writing). The live sherlock.wattson.it page still points at v1
   specifically — this script uses v1, matching what the dataset's own
   distribution page actually serves right now, rather than silently
   upgrading to a version the site itself hasn't linked. If sherlock.wattson.it
   later repoints to a newer record, this script's URLs/hashes will need
   updating and will fail loudly (md5 mismatch) rather than silently
   accepting different bytes under the old hash.

3. **Format is IPAL, not raw IEC-104.** Sherlock's own description states
   captures are "transcribed into the Industrial Protocol Abstraction
   Layer (IPAL) format for easy processing," alongside the raw network
   captures. `src/perception/sherlock_loader.py` parses the IPAL
   JSON-lines representation (gzip + stdlib `json`), not raw IEC-104
   packets — the tractable path the dataset's own authors built for
   exactly this purpose, and consistent with this project's
   no-new-dependency practice (Sessions 5–6): no `ipal_ids_framework`
   package is installed, IPAL's message/state schema (documented at
   https://github.com/ipal-ids/ipal) is small enough to parse directly.

4. **Exact internal zip layout — CONFIRMED after download, and it is NOT
   message-level IEC-104 traffic.** `01-Basic.zip` extracts to
   `01-Basic/01-Basic/` containing:
   ```
   train.n302.state.gz, test.n302.state.gz   # the real telemetry (below)
   ipal/{train,test}/events.json             # event catalog (start/end/id/description)
   ipal/{train,test}/initial_state.json      # single t=0 snapshot
   raw/{train,test}/pcap/                    # raw network captures (unparsed by this project)
   raw/{train,test}/log/*-service.log        # per-host service text logs
   raw/{train,test}/data-point-map.json      # IEC-104 point address -> {element, attribute, unit}
   raw/{train,test}/docs/{network,power-grid,combined}.{svg,pdf}  # topology diagrams (not machine-readable data)
   raw/{train,test}/control-center.zip       # nested archive, not opened this session
   ```
   `train.n302.state.gz`/`test.n302.state.gz` are gzipped JSON-lines, ONE
   PHYSICAL GRID STATE SNAPSHOT PER SECOND (verified: exactly 1.0s cadence,
   43204 lines each in 01-Basic) — see "Real state format" below. There is
   **no message-level `.ipal` export** (no `src`/`dest`/`activity` fields
   anywhere in what ships) — the task description's assumption of IEC-104
   message traffic transcribed into IPAL was wrong for what this scenario
   actually distributes; `data-point-map.json` maps each real protocol
   point address to a semantic name (e.g. `101.10010` -> `bus.0: voltage`),
   confirming the state export's `bus.N:voltage`-style keys ARE derived
   from the real IEC-104 addressing, just already resolved to semantic
   names rather than shipped as raw addresses.
   `src/perception/sherlock_loader.py` was rewritten around this real
   format; its original message-level design (host/IED classification, a
   comms-only asset graph) was removed rather than left as dead/speculative
   code once this was confirmed.

## Real state format (verified directly from the downloaded `01-Basic` scenario, 2026-08-05)

One JSON object per line in `{train,test}.n302.state.gz`:
```json
{"timestamp": 1741555756.202053,
 "state": {"bus.0:voltage": 1.0298183817271016, "bus.0:voltage_angle": -0.0005,
           "line.0:active_power_from": 3122710.586842443, "switch.8:closed": true,
           "load.0:active_power": 9955330.453309834, "trafo.0:tap_position": 0,
           "sgen.5:active_power": ...},
 "malicious": false}
```
470 keys per line (01-Basic scenario), spanning components `bus`/`line`/
`switch`/`load`/`trafo`/`sgen`/`external_grid`. `sgen` (pandapower's static
generator) is the closest analog to this project's DER concept.

`malicious` is **not boolean** — verified across both real files:
- `false` — nothing active (37,005 of 43,204 test lines)
- `"<n> (benign event)"` — a non-attack scripted event is active (e.g.
  `"27 (benign event)"`, 75 lines) — a NEGATIVE example, same as `false`
- a bare numeric string, e.g. `"14"` (595 lines) — a REAL ATTACK is active
  (event id `14`, cross-referenced against `ipal/test/events.json`)

The train file's `malicious` is `false` on 43,165/43,204 lines and
`"<n> (benign event)"` on the remaining 39 — **zero real attacks**,
confirming the task description's "two networks have both attack-free and
attack data" for the train/test pairing of 01-Basic. No separate
attack-interval file needs parsing: ground truth is already resolved
per-record.

**No verified electrical topology.** `data-point-map.json` names components
but never connects them (no `from_bus`/`to_bus` per line) — reconstructing
real connectivity would require parsing the SVG/PDF topology diagrams or
the raw pcaps, out of scope this session. `src/perception/sherlock_loader.py`
and `experiments/exp07_sherlock.py` therefore use a topology-FREE
`CausalTCN` classifier over an aggregate per-slice feature vector, not the
twin's HGTConv `PerceptionEncoder` pipeline — a stated architectural
divergence, not a silent downgrade.
