"""Inspect exact profiler TASK-to-CANN connection IDs, without span ownership."""
from pathlib import Path
import json, sqlite3

T = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
R = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O = T / 'graph-launch-link-inspection'
O.mkdir(exist_ok=False)
mapping = json.loads((T / 'captured-layer-attribution/module-call-metrics.json').read_text())
detail, summaries = [], []
for rp in sorted(R.glob('*_ascend_pt')):
    db = next((rp / 'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
    c = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)
    c.execute('PRAGMA query_only=ON'); c.execute('PRAGMA temp_store=MEMORY')
    mods = [m for m in mapping if m['rank'] == rp.name and m['layer'] == 0]
    joined, absent = 0, 0
    for m in sorted(mods, key=lambda m: m['window_start_ns']):
        rows = list(c.execute('''SELECT t.startNs,t.endNs,t.connectionId,t.globalTaskId,t.taskId,
             a.startNs,a.endNs,s.value FROM TASK t
             LEFT JOIN CANN_API a ON t.connectionId=a.connectionId
             LEFT JOIN STRING_IDS s ON a.name=s.id
             WHERE t.modelId=? AND t.streamId=? AND t.taskId=?
             AND t.startNs>=? AND t.startNs<=?''',
             (int(m['model_id']), int(m['main_stream_id']), int(m['marker_task_ids'][0]),
              m['window_start_ns'] - 1000, m['window_start_ns'] + 1000)))
        if not rows:
            absent += 1
        for row in rows:
            if row[5] is not None:
                joined += 1
            detail.append({'rank': rp.name, 'captured_run_in_model': m['run_in_model'],
                           'layer_start_ns': m['window_start_ns'], 'task_start_ns': row[0], 'task_end_ns': row[1],
                           'connection_id': row[2], 'global_task_id': row[3], 'task_id': row[4],
                           'cann_start_ns': row[5], 'cann_end_ns': row[6], 'cann_name': row[7],
                           'task_after_cann_end_us': (row[0] - row[6]) / 1000 if row[6] is not None else None})
    names = list(c.execute('''SELECT s.value,count(*) FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id
                  WHERE lower(s.value) LIKE '%model%' OR lower(s.value) LIKE '%graph%'
                  GROUP BY a.name ORDER BY count(*) DESC LIMIT 15'''))
    summaries.append({'rank': rp.name, 'layer0_calls': len(mods), 'task_exact_query_absent': absent,
                      'joined_cann_records': joined, 'graph_model_cann_names': names})
    c.close()
(O / 'connections.json').write_text(json.dumps(detail, indent=2) + '\n')
doc = {'status': 'READ_ONLY_EXACT_CONNECTION_INSPECTION', 'ranks': summaries,
       'no_device_work': True, 'no_CPU_span_containment_ownership': True,
       'sample_connections_first_2_ranks': [r for r in detail if r['rank'] in [summaries[0]['rank'], summaries[8]['rank']]],
       'limitations': 'Connection IDs are exporter-reported. A connection must be validated as model replay submission before interpreting host/device queue delay.'}
assert len(json.dumps(doc).encode()) < 32768
(O / 'summary.json').write_text(json.dumps(doc, indent=2) + '\n')
print(json.dumps(doc, indent=2))
