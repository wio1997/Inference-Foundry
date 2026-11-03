#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/next-selective-tests/gate.XXXXXX")
TMPDIR="$run";export TMPDIR
ulimit -c 0
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_recorder_next_gate.c" -o "$run/libselective.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/selective_gate_mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/selective_gate_test.c" -L"$run" -lmock -o "$run/test" -pthread
mkdir "$run/normal" "$run/concurrent"
for mode in normal unknown concurrent;do
 case "$mode" in unknown) dir="$run/missing";; *) dir="$run/$mode";; esac
 if [ "$mode" = concurrent ];then touch "$dir/PROFILE_ACTIVE";fi
 LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libselective.so" BINDING_TRACE_PATH="$run/$mode.jsonl" WIO_DIAG_DIR="$dir" "$run/test" "$mode" "$dir/PROFILE_ACTIVE"
done
python3 - "$run" <<'PY2'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
def rows(n):return [json.loads(x) for x in (p/(n+'.jsonl')).read_text().splitlines()]
a=rows('normal');assert sum(x['event']=='target_launch_begin' for x in a)==3
assert len((p/'normal.jsonl').read_bytes())<20000
assert all(x['pre_call_sha256']=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' for x in a if x['event']=='target_launch_begin')
b=rows('unknown');assert sum(x['event']=='gate_unknown' for x in b)==1 and not any(x['event']=='target_launch_begin' for x in b)
s=next(x for x in b if x['event']=='selective_stats' and x['reason']=='process_exit');assert s['gate_unknown']>0 and s['state_incomplete']
c=rows('concurrent');s=next(x for x in c if x['event']=='selective_stats' and x['reason']=='process_exit');assert s['launch_mutation_overlaps']>0 and s['state_incomplete'],s
print(json.dumps({'status':'PASS','normal_bytes':(p/'normal.jsonl').stat().st_size,'warmup_target_launches':12000,'active_target_launches':3,'unknown_gate_count':len(b),'concurrent_overlaps':s['launch_mutation_overlaps'],'result_dir':str(p)}))
PY2
