"""CPU-only audit of the denominators used by the module counter proxy."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from collections import defaultdict
import csv, json, sqlite3

T = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
R = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
out = T / 'counter-field-audit'
out.mkdir(exist_ok=False)
summary = []
bad_examples = []
def dec(s):
    try:
        n = Decimal(s)
        return n if n.is_finite() else None
    except (InvalidOperation, TypeError):
        return None
for rp in sorted(R.glob('*_ascend_pt')):
    counts = defaultdict(int)
    for row in csv.DictReader((rp / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
        if row['Model ID'] in ['4294967295', 'N/A', ''] or 'metadata' in row['Type'].lower():
            continue
        counts['captured_nonmetadata_rows'] += 1
        for engine, ratio in [('aic', 'mac'), ('aiv', 'vec')]:
            cycles = dec(row[engine + '_total_cycles'])
            activity = dec(row[engine + '_' + ratio + '_ratio'])
            if cycles is not None and cycles > 0:
                counts[engine + '_positive_cycle_rows'] += 1
                if activity is None or not 0 <= activity <= 1:
                    counts[engine + '_invalid_activity_with_positive_cycles'] += 1
                    if len(bad_examples) < 20:
                        bad_examples.append({'rank': rp.name, 'type': row['Type'], 'task': row['Task ID'], 'engine': engine})
            elif cycles is None:
                counts[engine + '_missing_cycle_rows'] += 1
                if activity is not None and activity > 0:
                    counts[engine + '_positive_activity_missing_cycles'] += 1
    summary.append({'rank': rp.name, **counts})
status = 'PASS_NO_POSITIVE_CYCLE_ROWS_WITH_MISSING_OR_INVALID_RATIOS'
if bad_examples:
    status = 'FAIL_REVIEW_MODULE_COUNTER_RANKING'
doc = {'status': status, 'ranks': summary, 'bad_examples': bad_examples,
       'limitations': 'CSV report consistency only; does not validate vendor PMU semantics, exact activity aggregation, clock conversion, or profiling overhead.',
       'no_device_work': True}
(out / 'summary.json').write_text(json.dumps(doc, indent=2) + '\n')
rp = next(p for p in sorted(R.glob('dp0*_ascend_pt')) if '_tp0_' in p.name)
db = next((rp / 'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
c = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)
c.execute('PRAGMA query_only=ON'); c.execute('PRAGMA temp_store=MEMORY')
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if any(x in r[0].upper() for x in ['API', 'STRING', 'TASK', 'CONNECTION'])]
schemas = {n: list(c.execute('PRAGMA table_info("' + n.replace('"', '""') + '")')) for n in tables}
(out / 'db-schema.json').write_text(json.dumps({'db': str(db), 'schemas': schemas}, indent=2) + '\n')
print(json.dumps({'status': status, 'ranks': len(summary), 'bad_examples': bad_examples, 'db_tables': tables}, indent=2))
assert not bad_examples
