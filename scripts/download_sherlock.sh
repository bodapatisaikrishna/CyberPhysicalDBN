#!/usr/bin/env bash
# Download the Sherlock power-grid IDS dataset (Wagner, Bader, Wolsing,
# Serror; "Sherlock: A Dataset for Process-aware Intrusion Detection
# Research on Power Grid Networks," ACM CODASPY'25) from Zenodo.
#
# Full details, discrepancies against this project's own task description,
# and the exact IPAL schema this dataset uses: docs/sherlock_download.md.
#
# NEVER invoked automatically by any experiment or test in this repo
# (CLAUDE.md rule 1 / "downloading a file" is an explicit-permission action)
# -- a human, or an agent that has just received explicit chat permission
# stating the exact filename/source/size, must run this by hand.
#
# Usage:
#   scripts/download_sherlock.sh [--scenario 01-Basic|02-Semiurban|03-Rural|all] [--out DIR] [--with-paper]
#
# Default --scenario is 01-Basic (704.1 MB) -- the only scenario small
# enough to be a responsible default. 02-Semiurban (4.7 GB) and 03-Rural
# (1.9 GB) require explicit opt-in and print a size warning before
# fetching anything.

set -euo pipefail

# --- Zenodo record 15168928 (v1, published 2025-04-10) ----------------------
# The live https://sherlock.wattson.it/download/ page links to this exact
# record. Zenodo itself reports newer versions exist (v2: 15260901,
# v3: 18467070, latest as of this writing 2026-02-04) but the site has not
# repointed its own link -- v1 is used here as what the dataset's own
# distribution page actually serves, not silently upgraded to a newer
# version. Re-check https://sherlock.wattson.it/download/ before reusing
# this script much later.
ZENODO_BASE="https://zenodo.org/records/15168928/files"
PAPER_URL="${ZENODO_BASE}/paper.pdf"
PAPER_MD5="662db881140984b51952d674daac4a25"

# Plain case statements, not `declare -A`: macOS ships bash 3.2 by default
# (verified in this environment), which predates bash 4.0's associative
# arrays entirely -- `declare -A` fails there even under `#!/usr/bin/env
# bash`, since there is no newer bash installed to pick up instead.
sherlock_url() {
  case "$1" in
    01-Basic) echo "${ZENODO_BASE}/01-Basic.zip" ;;
    02-Semiurban) echo "${ZENODO_BASE}/02-Semiurban.zip" ;;
    03-Rural) echo "${ZENODO_BASE}/03-Rural.zip" ;;
  esac
}
sherlock_md5() {
  case "$1" in
    01-Basic) echo "4f751246a245b952f0200e74ef1da10f" ;;
    02-Semiurban) echo "e864944c52fb4a6f27b544c08a351ae7" ;;
    03-Rural) echo "2925a5275ef63d9a413a218fc667fd44" ;;
  esac
}
sherlock_size() {
  case "$1" in
    01-Basic) echo "704.1 MB" ;;
    02-Semiurban) echo "4.7 GB" ;;
    03-Rural) echo "1.9 GB" ;;
  esac
}

SCENARIO="01-Basic"
OUT_DIR="data/sherlock"
WITH_PAPER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --with-paper) WITH_PAPER=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

fetch_and_verify() {
  local name="$1" url="$2" expected_md5="$3" dest_dir="$4"
  local zip_path="${dest_dir}/${name}.zip"

  mkdir -p "$dest_dir"
  echo "==> fetching ${name} (${url})"
  curl --fail --location --output "${zip_path}" "${url}"

  echo "==> verifying md5 for ${name}"
  local actual_md5
  if command -v md5sum >/dev/null 2>&1; then
    actual_md5="$(md5sum "${zip_path}" | awk '{print $1}')"
  else
    # macOS has no md5sum by default; md5 -q prints just the hash.
    actual_md5="$(md5 -q "${zip_path}")"
  fi
  if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    echo "MD5 MISMATCH for ${name}: expected ${expected_md5}, got ${actual_md5}" >&2
    echo "The downloaded file does not match the hash this script was written against." >&2
    echo "Do NOT proceed as if it does; Zenodo's file may have changed, or the download was corrupted." >&2
    exit 1
  fi

  echo "==> unzipping ${name} into ${dest_dir}/${name}/"
  mkdir -p "${dest_dir}/${name}"
  unzip -q "${zip_path}" -d "${dest_dir}/${name}"
  rm "${zip_path}"
  echo "==> ${name} ready at ${dest_dir}/${name}/"
}

download_scenario() {
  local name="$1"
  echo
  echo "Scenario ${name}: $(sherlock_size "${name}")"
  if [[ "${name}" != "01-Basic" ]]; then
    echo "WARNING: ${name} is $(sherlock_size "${name}") -- significantly larger than 01-Basic (704.1 MB)."
    echo "         Confirm this is intentional before it starts downloading."
  fi
  fetch_and_verify "${name}" "$(sherlock_url "${name}")" "$(sherlock_md5 "${name}")" "${OUT_DIR}"
}

case "${SCENARIO}" in
  01-Basic|02-Semiurban|03-Rural)
    download_scenario "${SCENARIO}"
    ;;
  all)
    for name in 01-Basic 02-Semiurban 03-Rural; do
      download_scenario "${name}"
    done
    ;;
  *)
    echo "unknown --scenario '${SCENARIO}': expected 01-Basic, 02-Semiurban, 03-Rural, or all" >&2
    exit 1
    ;;
esac

if [[ "${WITH_PAPER}" -eq 1 ]]; then
  echo
  echo "==> fetching paper.pdf"
  mkdir -p "${OUT_DIR}"
  curl --fail --location --output "${OUT_DIR}/paper.pdf" "${PAPER_URL}"
  actual_md5="$( (command -v md5sum >/dev/null 2>&1 && md5sum "${OUT_DIR}/paper.pdf" | awk '{print $1}') || md5 -q "${OUT_DIR}/paper.pdf")"
  if [[ "${actual_md5}" != "${PAPER_MD5}" ]]; then
    echo "MD5 MISMATCH for paper.pdf: expected ${PAPER_MD5}, got ${actual_md5}" >&2
    exit 1
  fi
fi

echo
echo "Done. See docs/sherlock_download.md for dataset notes and known"
echo "discrepancies against this project's own task description."
