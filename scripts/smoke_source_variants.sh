#!/bin/bash
# Smoke-test the pipeline local / copick-local and the ctf-refine / polish portal variants end to end,
# without GPUs and without running RELION or submitting to SLURM: the pipeline variants stop at
# --prepare-only, the ctf-refine / polish variants at --dry-run. Nothing is mocked, so the portal-backed
# cases exercise real portal queries, star-file generation and path resolution.
#
# Not covered here: extract, reconstruct, tomograms, export, pipeline data-portal /
# copick-data-portal, and ctfrefine / polish local -- those are covered by the pytest suite.
#
# Everything is written under a temp dir that is removed on exit (pass KEEP=1 to keep it).
#
# Requires: this package importable, py2rely + RELION binaries on PATH (for the pipeline variants'
# preflight), copick, and network access to the CryoET Data Portal.
#
# Usage:
#   scripts/smoke_source_variants.sh --tomograms-star <star> --particles-star <star> \
#       --template <ref.mrc> [--copick-config <cfg.json> --copick-name N --copick-user U --copick-session S]
set -o pipefail

TOMOGRAMS_STAR="" PARTICLES_STAR="" TEMPLATE=""
COPICK_CONFIG="" COPICK_NAME="" COPICK_USER="" COPICK_SESSION=""
RUN_IDS="16848"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tomograms-star) TOMOGRAMS_STAR="$2"; shift 2 ;;
    --particles-star) PARTICLES_STAR="$2"; shift 2 ;;
    --template)       TEMPLATE="$2";       shift 2 ;;
    --copick-config)  COPICK_CONFIG="$2";  shift 2 ;;
    --copick-name)    COPICK_NAME="$2";    shift 2 ;;
    --copick-user)    COPICK_USER="$2";    shift 2 ;;
    --copick-session) COPICK_SESSION="$2"; shift 2 ;;
    --run-ids)        RUN_IDS="$2";        shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for req in TOMOGRAMS_STAR PARTICLES_STAR TEMPLATE; do
  [[ -n "${!req}" ]] || { echo "missing --${req,,} (see header)" | tr '_' '-' >&2; exit 2; }
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/zpt-smoke-XXXXXX")
cleanup() { if [[ "${KEEP:-0}" == "1" ]]; then echo "kept $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT
echo "workdir: $WORK"

PASS=0 FAIL=0
declare -a RESULTS
check() {
  local name="$1"; shift
  if "$@" > "$WORK/$name.log" 2>&1; then
    RESULTS+=("PASS  $name"); PASS=$((PASS + 1))
  else
    RESULTS+=("FAIL  $name  (see $WORK/$name.log)"); FAIL=$((FAIL + 1))
    tail -5 "$WORK/$name.log" | sed 's/^/        /'
  fi
}

# ---- pipeline local: stars must live inside --output-dir, so stage them there
P_LOCAL=$WORK/pipeline_local
mkdir -p "$P_LOCAL/input"
cp "$TOMOGRAMS_STAR" "$P_LOCAL/input/tomograms.star"
cp "$PARTICLES_STAR" "$P_LOCAL/input/particles.star"
[[ -d "$(dirname "$TOMOGRAMS_STAR")/tiltseries" ]] && cp -r "$(dirname "$TOMOGRAMS_STAR")/tiltseries" "$P_LOCAL/input/"
check pipeline_local zarr-particle-pipeline local \
  --output-dir "$P_LOCAL" \
  --particles-starfile "$P_LOCAL/input/particles.star" \
  --tomograms-starfile "$P_LOCAL/input/tomograms.star" \
  --protein-diameter 330 --reference-template "$TEMPLATE" --prepare-only

# ---- ctf-refine / polish portal variants: --dry-run stops after the tomograms.star
touch "$WORK/half1.mrc" "$WORK/half2.mrc"
for tool in ctfrefine polish; do
  check "${tool}_data_portal" "zarr-particle-$tool" data-portal \
    --run-ids "$RUN_IDS" --annotation-names ribosome --inexact-match --ground-truth \
    --particles-starfile "$PARTICLES_STAR" \
    --ref1 "$WORK/half1.mrc" --ref2 "$WORK/half2.mrc" \
    --box-size 64 --output-dir "$WORK/${tool}_dp" --dry-run
done

# ---- copick-backed variants, only if a copick project was supplied
if [[ -n "$COPICK_CONFIG" ]]; then
  P_CL=$WORK/pipeline_copick_local
  mkdir -p "$P_CL/input"
  cp "$TOMOGRAMS_STAR" "$P_CL/input/tomograms.star"
  [[ -d "$(dirname "$TOMOGRAMS_STAR")/tiltseries" ]] && cp -r "$(dirname "$TOMOGRAMS_STAR")/tiltseries" "$P_CL/input/"
  check pipeline_copick_local zarr-particle-pipeline copick-local \
    --output-dir "$P_CL" --tomograms-starfile "$P_CL/input/tomograms.star" \
    --copick-config "$COPICK_CONFIG" --copick-name "$COPICK_NAME" \
    --copick-user-id "$COPICK_USER" --copick-session-id "$COPICK_SESSION" \
    --protein-diameter 330 --reference-template "$TEMPLATE" --prepare-only

  for tool in ctfrefine polish; do
    check "${tool}_copick_data_portal" "zarr-particle-$tool" copick-data-portal \
      --copick-config "$COPICK_CONFIG" --copick-name "$COPICK_NAME" \
      --copick-user-id "$COPICK_USER" --copick-session-id "$COPICK_SESSION" \
      --particles-starfile "$PARTICLES_STAR" \
      --ref1 "$WORK/half1.mrc" --ref2 "$WORK/half2.mrc" \
      --box-size 64 --output-dir "$WORK/${tool}_cdp" --dry-run
  done
else
  echo "note: no --copick-config given, skipping the 3 copick-backed variants"
fi

echo
printf '%s\n' "${RESULTS[@]}"
echo "passed=$PASS failed=$FAIL"
[[ $FAIL -eq 0 ]]
