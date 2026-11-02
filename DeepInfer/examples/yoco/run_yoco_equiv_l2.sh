#!/usr/bin/env bash
set -euo pipefail

mkdir -p var
run_dir=$(mktemp -d var/yoco_equiv.XXXXXX)
mask_out="$run_dir/logp_mask.json"
compact_out="$run_dir/logp_compact.json"

python examples/yoco/yoco_equiv.py logp "$@" --compact 0 --out "$mask_out"
python examples/yoco/yoco_equiv.py logp "$@" --compact 1 --out "$compact_out"
python examples/yoco/yoco_equiv.py compare "$mask_out" "$compact_out"

echo "outputs: $run_dir"
