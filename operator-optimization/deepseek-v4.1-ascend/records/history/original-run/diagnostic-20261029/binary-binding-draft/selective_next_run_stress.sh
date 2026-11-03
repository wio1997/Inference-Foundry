#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/next-selective-tests/stress.XXXXXX")
TMPDIR="$run";export TMPDIR
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -DBIN_SLOTS=4 -DFUNC_SLOTS=8 -DPENDING_SLOTS=4 -DPENDING_BATCH=2 -DBINDING_MAX_LOG_BYTES=4096 -I"$inc" "$here/selective_recorder_next.c" -o "$run/libselective.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_next_mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/selective_next_stress.c" -L"$run" -lmock -o "$run/test" -pthread
for mode in evict cap missing pending;do
 LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libselective.so" BINDING_TRACE_PATH="$run/$mode.jsonl" "$run/test" "$mode"
 python3 "$here/selective_next_stress_validate.py" "$run/$mode.jsonl" "$mode"
done
printf 'CPU-only stress directory: %s\n' "$run"
