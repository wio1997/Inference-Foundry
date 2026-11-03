#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run=$(mktemp -d "$here/mock-cap.XXXXXX")
TMPDIR="$run"; export TMPDIR
inc=/usr/local/Ascend/cann-9.2.0/aarch64-linux/include
gcc -Wall -Wextra -Werror -fPIC -shared -DBINDING_MAX_LOG_BYTES=4096 -I"$inc" "$here/recorder.c" -o "$run/libbinding_recorder.so" -ldl -pthread
gcc -Wall -Wextra -Werror -fPIC -shared -I"$inc" "$here/mock.c" -o "$run/libmock.so"
gcc -Wall -Wextra -Werror -I"$inc" "$here/test.c" -L"$run" -lmock -o "$run/test_mock" -pthread
printf abc > "$run/sample.bin"
for i in 1 2 3 4; do
 LD_LIBRARY_PATH="$run" LD_PRELOAD="$run/libbinding_recorder.so" BINDING_TRACE_PATH="$run/trace.jsonl" "$run/test_mock" "$run/sample.bin" > "$run/worker-$i.txt" 2>&1 &
done
wait
for i in 1 2 3 4; do grep -q "mock forwarding PASS" "$run/worker-$i.txt"; done
python3 - "$run/trace.jsonl" <<'PY' | tee "$run/result.json"
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); raw=p.read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
assert len(raw)<=4096, len(raw)
assert len(rows)>1 and rows[-1]['event']=='trace_cap', rows[-2:]
assert sum(r['event']=='trace_cap' for r in rows)==1
assert rows[-1]['max_bytes']==4096
print(json.dumps({'status':'PASS','scope':'CPU only; four independent writer processes plus fork; cooperating writers','size':len(raw),'cap':4096,'rows':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'marker':'present and last'},indent=2))
PY
printf 'CPU-only cap result path: %s\n' "$run/result.json"
