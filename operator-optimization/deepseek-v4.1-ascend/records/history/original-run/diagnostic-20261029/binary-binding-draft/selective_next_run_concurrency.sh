#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/next-selective-tests/concurrency.XXXXXX")
TMPDIR="$run";export TMPDIR
ulimit -c 0
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_recorder_next.c" -o "$run/libselective.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_next_mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/selective_next_concurrency.c" -L"$run" -lmock -o "$run/test" -pthread
LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libselective.so" BINDING_TRACE_PATH="$run/trace.jsonl" "$run/test"
python3 - "$run/trace.jsonl" <<'PY'
import json,sys
r=[json.loads(x) for x in open(sys.argv[1])];s=next(x for x in r if x['event']=='selective_stats' and x['reason']=='process_exit')
assert s['mutation_overlaps']>0 and s['state_incomplete'],s
assert all(x.get('binding_status')=='candidate_unverified' for x in r if x['event'] in ('target_function','target_load','target_launch_begin','target_byentry_reclassified','target_preclassification_launch'))
print(json.dumps({'mode':'concurrent_mutation','status':'PASS','stats':s}))
PY
