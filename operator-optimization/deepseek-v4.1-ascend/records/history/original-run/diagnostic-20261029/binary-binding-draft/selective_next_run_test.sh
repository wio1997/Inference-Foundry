#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/next-selective-tests/run.XXXXXX")
TMPDIR="$run"; export TMPDIR
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_recorder_next.c" -o "$run/libselective.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_next_mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/selective_next_test.c" -L"$run" -lmock -o "$run/test" -pthread
obj=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/dispatch_ffn_combine/DispatchFFNCombine_17df546b46e1ea56007584fac7e6f429.o
LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libselective.so" BINDING_TRACE_PATH="$run/trace.jsonl" "$run/test" "$obj"
python3 "$here/selective_next_validate.py" "$run/trace.jsonl" | tee "$run/result.json"
printf 'CPU-only result: %s\n' "$run/result.json"
