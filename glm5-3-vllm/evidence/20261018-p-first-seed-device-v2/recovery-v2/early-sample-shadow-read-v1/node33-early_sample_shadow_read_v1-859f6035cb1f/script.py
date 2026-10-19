from pathlib import Path
from collections import Counter
import datetime
import json
import socket
import statistics
import subprocess

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-copy-shadow-live1-20261019/node33')
cid='9b788715cb6e34139c2e9a70ef5bcf49009dda507164bf11e9e00cd07c163863'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
assert d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
probe=root/'early-sample-copy-shadow';assert not(probe/'enable').exists()
rows=[json.loads(s) for p in sorted(probe.glob('rank-*.jsonl')) for s in p.read_text().splitlines()]
assert rows and sorted({z['rank'] for z in rows})==list(range(16))
assert all(z['bitwise_equal'] and z['native_event_kept'] and not z['early_publication'] and z['source_keepalive_until_native_completion'] for z in rows)
by_rank={str(n):sum(z['rank']==n for z in rows) for n in range(16)};assert min(by_rank.values())>=5
wait=sorted(z['native_wait_parse_us'] for z in rows)
report=dict(completed=True,CID=cid,observed_ranks=list(range(16)),rows=len(rows),rows_by_rank=by_rank,
            bitwise_all=True,native_event_kept_all=True,early_publication=False,source_kept_until_native_completion_all=True,
            shapes=dict(Counter(str(z['shape']) for z in rows)),
            early_ready_at_late_constructor=sum(z['early_ready_at_late_constructor'] for z in rows),
            early_ready_at_native_get=sum(z['early_ready_at_native_get'] for z in rows),
            early_ready_while_late_not_ready_at_native_get=sum(z['early_ready_at_native_get'] and not z['late_ready_at_native_get'] for z in rows),
            native_wait_parse_us=dict(min=wait[0],median=statistics.median(wait),max=wait[-1]),
            remaining_model_s=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds(),
            scope='Instrumented native PD requests. CPU readiness queries observe instants only; wait/parse spans are diagnostic, not isolated MTP time or service gain. Both native late copy/event and InputBatch readiness are unchanged. Bitwise shadow proof is finite, not general quality or safe early retirement proof.')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
