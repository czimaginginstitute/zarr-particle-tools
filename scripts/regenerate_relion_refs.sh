#!/usr/bin/env bash
# Regenerate RELION 5 reconstruct references on the HPC with the pinned RELION commit.
#
# RELION reference version: daniel-ji/relion master @ b1fe45f6 ("5.1.0-commit-b1fe45"),
#   built under: gcc/13.3, cuda/12.8.0_570.86.10, openmpi/5.0.7-cuda12.8, cmake/3.28.2
#   binaries: /home/daniel.ji/work/relion/build/bin
#
# Usage:  scripts/regenerate_relion_refs.sh <SYM> <out_dir> [box_size] [bin]
# Example (I2 reference for Fix E):
#   scripts/regenerate_relion_refs.sh I2 /tmp/relion_ref_I2 64 1
set -euo pipefail

SYM="${1:?symmetry, e.g. C1 or I2}"
OUT="$(readlink -f "${2:?output dir}")"
BOX="${3:-64}"
BIN="${4:-1}"
# Repo root (this script lives in <repo>/scripts/); RELION resolves the tomogram-set's
# rlnTomoTiltSeriesStarFile paths (e.g. tiltseries/TS_1.star) relative to CWD, so we must
# run from inside the RELION project directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO/tests/data/relion_project_synthetic"

source /hpc/modules/lmod/8.7.59/libexec/lmod/init/bash 2>/dev/null
module load gcc/13.3 cuda/12.8.0_570.86.10 openmpi/5.0.7-cuda12.8
export PATH="/home/daniel.ji/work/relion/build/bin:$PATH"

cd "$DATA"
relion_tomo_reconstruct_particle \
  --p particles.star \
  --t tomograms.star \
  --b "$BOX" \
  --bin "$BIN" \
  --sym "$SYM" \
  --o "$OUT/" \
  --j 16

echo "RELION reconstruct done: $OUT/merged.mrc"
