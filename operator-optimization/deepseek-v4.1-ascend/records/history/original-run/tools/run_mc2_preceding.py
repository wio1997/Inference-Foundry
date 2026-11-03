from pathlib import Path
import subprocess, shlex

T = Path(__file__).resolve().parents[1]
R = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C = 'wio-vllm-v41-repro169-stage-20261028'
src = T / 'tools/analyze_mc2_preceding.py'
code = 'from pathlib import Path;p=Path(' + repr(R + '/tools/' + src.name) + ');assert not p.exists();p.write_text(' + repr(src.read_text()) + ')'
r = subprocess.run([SSH, shlex.join(['docker', 'exec', C, 'python3', '-c', code])], capture_output=True, text=True, timeout=90)
assert r.returncode == 0, r.stderr
r = subprocess.run([SSH, shlex.join(['docker', 'exec', C, 'python3', R + '/tools/' + src.name])], timeout=240)
assert r.returncode == 0, r.returncode
for name in ['summary.json', 'rank_layer0_timeline.csv', 'paired_timeline.csv', 'bounded_local_slices.csv']:
    code = 'from pathlib import Path;p=Path(' + repr(R + '/mc2-preceding-analysis/' + name) + ');assert p.stat().st_size<2*1024*1024;print(p.read_text(),end="")'
    r = subprocess.run([SSH, shlex.join(['docker', 'exec', C, 'python3', '-c', code])], capture_output=True, timeout=90)
    assert r.returncode == 0, r.stderr.decode()
    p = T / 'mc2-preceding-analysis' / name
    p.parent.mkdir(exist_ok=True); p.write_bytes(r.stdout)
