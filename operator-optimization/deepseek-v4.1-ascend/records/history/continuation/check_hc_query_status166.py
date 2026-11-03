from pathlib import Path
from datetime import datetime, timezone
import json, os, socket, signal

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
out = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-hc-boundary-task-stack-evidence.json')
status = dict(utc=datetime.now(timezone.utc).isoformat(), result_exists=out.exists(), own_query_processes=[])
if out.exists():
    d = json.loads(out.read_text())
    status.update(database_sha256=d['database_sha256'], database_stat_unchanged=d['database_stat_unchanged'],
                  full_result=str(out), item_summaries=[{k:x[k] for k in ['label','interval_ns','record_count','compute_records','op_type_counts']} for x in d['items']])
else:
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    hz = os.sysconf('SC_CLK_TCK')
    marker = b"out.write_text(json.dumps(result, ensure_ascii=False, indent=2)"
    filename = out.name.encode()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            cmd = (p/'cmdline').read_bytes()
            if not (cmd.startswith(b'python3\x00-c\x00') and marker in cmd and filename in cmd):
                continue
            stat = (p/'stat').read_text().rsplit(')',1)[1].split()
            elapsed = uptime - int(stat[19])/hz
            row = dict(pid=int(p.name), elapsed_seconds=elapsed, stopped=False)
            if elapsed >= 180 and (p/'cmdline').read_bytes() == cmd:
                os.kill(int(p.name), signal.SIGTERM)
                row['stopped'] = True
            status['own_query_processes'].append(row)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
print(json.dumps(status, ensure_ascii=False))
