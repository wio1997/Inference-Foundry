#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/mock-run.XXXXXX")
TMPDIR="$run"; export TMPDIR
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/recorder.c" -o "$run/libbinding_recorder.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/test.c" -L"$run" -lmock -o "$run/test_mock" -pthread
printf abc > "$run/sample.bin"
LD_LIBRARY_PATH="$run" "$run/test_mock" "$run/sample.bin"
LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libbinding_recorder.so" BINDING_TRACE_PATH="$run/trace.jsonl" "$run/test_mock" "$run/sample.bin"
python3 "$here/validate_mock.py" "$run/trace.jsonl" | tee "$run/result.json"
printf 'CPU-only result path: %s\n' "$run/result.json"
