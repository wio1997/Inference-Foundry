"""Retarget existing future-launch defaults only; never load weights or run a service."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

cfg = json.loads(sys.argv[1])
D = Path('/data/tiankuan/wio/glm52-pd/deploy')
backup = D/'private'/'GLM53-CONFIG-20261006'/'originals'
assert not backup.exists(), 'Existing attempt must be preserved, not repeated'
replacements = {
    '/public-flash/models/GLM-5.2-w8a8': '/data/tiankuan/wio/GLM-5.3-w8a8',
    '/data/tiankuan/wio/GLM-5.2-w8a8': '/data/tiankuan/wio/GLM-5.3-w8a8',
    'glm-52': 'glm-53',
    'GLM-5.2 PD': 'GLM-5.3 PD',
}
notice = ('当前目标：GLM-5.3 W8A8标准P/D分离；两机权重'
          '`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务名`glm-53`。'
          '2026-10-06用户上传中，完成并核验config/index/tokenizer/全部分片前不加载或测试。'
          'glm52-single与glm52-pd/deploy是实际资源名；旧驻留服务和历史Run不改标为5.3。'
          '所列旧布局/参数/兼容修复仅作参考，5.3的fit、功能及PD性能尚未验收。'
          '研究方案以Inference-Foundry研究分支glm5-3/PLAN.md为准。')
planned = []
for name, expected in cfg['pins'].items():
    assert name in cfg['allowed_paths']
    p = D/name
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, f'Concurrent change: {name}'
    text = raw.decode()
    if name.endswith('.md'):
        # HANDOFF and START_HERE contain dated history: preserve their body.
        if name not in {'HANDOFF.md', 'START_HERE.md'}:
            for old, new in replacements.items():
                text = text.replace(old, new)
        text = '> GLM53_TARGET_20261006: '+notice+'\n\n'+text
    else:
        for old, new in replacements.items():
            text = text.replace(old, new)
        if p.suffix == '.sh':
            lines = text.splitlines(keepends=True)
            lines.insert(1, '# GLM-5.3 target defaults only; upload/fit/correctness pending. Do not execute during upload.\n')
            text = ''.join(lines)
    after = text.encode()
    if p.suffix == '.sh':
        subprocess.run(['bash', '-n'], input=after, check=True, capture_output=True)
    elif p.suffix == '.py':
        ast.parse(text, filename=str(p))
    planned.append((name, p, raw, after, p.stat()))
backup.mkdir(parents=True)
rows = []
for name, p, raw, after, st in planned:
    assert p.read_bytes() == raw, f'Concurrent change before write: {name}'
    saved = backup/name
    saved.parent.mkdir(parents=True, exist_ok=True)
    with saved.open('xb') as f:
        f.write(raw)
    temporary = p.with_name(p.name+'.GLM53-CONFIG-20261006.tmp')
    with temporary.open('xb') as f:
        f.write(after)
    os.chmod(temporary, st.st_mode)
    os.chown(temporary, st.st_uid, st.st_gid)
    os.replace(temporary, p)
    assert p.read_bytes() == after and saved.read_bytes() == raw
    rows.append(dict(path=str(p), before_sha256=hashlib.sha256(raw).hexdigest(),
                     after_sha256=hashlib.sha256(after).hexdigest(),
                     original_backup=str(saved), bytes=len(after)))
print(json.dumps(dict(host=cfg['host'], files=rows, syntax_passed=True,
                     target_model='/data/tiankuan/wio/GLM-5.3-w8a8',
                     served_model_name='glm-53', weight_payload_reads=0,
                     service_operations=0, inference_requests=0)))
