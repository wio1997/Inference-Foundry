#!/usr/bin/env python3
"""Validate the checkpoint's identities and arithmetic, not whole-task completion."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import traceback

ROOT=Path(__file__).resolve().parent
read=lambda path:json.loads((ROOT/path).read_text())
record={'scope':'source IDs/hashes, branch-review references, measured family arithmetic and open gates; no completeness proof'}
try:
    files=read('coverage-ledger/source-files.json');byfile={x['path']:x for x in files}
    assert len(files)==len(byfile)==558
    for x in files:assert hashlib.sha256(Path(x['local_path']).read_bytes()).hexdigest()==x['sha256']
    id_counts={};bybranch={}
    for kind in ('source-units','branch-sites','call-sites','import-sites'):
        items=read('coverage-ledger/'+kind+'.json');ids={x['id'] for x in items}
        assert len(ids)==len(items),(kind,len(items),len(ids))
        assert all(x['source'] in byfile and x['source_sha256']==byfile[x['source']]['sha256'] for x in items)
        id_counts[kind]=len(ids)
        if kind=='branch-sites':bybranch={x['id']:x for x in items}
    assert read('coverage-ledger/summary.json')['parse_errors']==[]
    module=read('module-coverage-ledger.json');assert not module['goal_scope_closed'] and not module['overall_complete']
    assert len(module['surfaces'])==26 and len(module['rule_coverage'])==260
    for surface in module['surfaces']:
        assert not surface['complete'] and len(surface['mandatory_gates'])==8
        assert all(g['status'] not in ('PASS','CLOSED','COMPLETE') for g in surface['mandatory_gates'].values())
        assert all(x in bybranch for x in surface['indexed_branch_ids'])
    delta=read('tail-branch-review-delta.json')['reviews']
    assert len(delta)==9 and len({x['branch_id'] for x in delta})==9
    for x in delta:
        original=bybranch[x['branch_id']]
        assert all(x[k]==original[k] for k in ('source','source_sha256','line','condition'))
        assert not x['whole_module_closed'] and not x['all_consumers_closed']
    exports=read('head-sampler-exports.json');summary=read('head-sampler-family-summary.json')
    assert {x['rank'] for x in exports['rows']}==set(range(16))
    old=json.loads((ROOT.parent/'diagnostic-20261029/official-analysis/operators.json').read_text())
    oldhash={x['source_dir']:x['sha256']['kernel_details.csv'] for x in old['rows']}
    for x in exports['rows']:
        assert x['csv_sha256']==oldhash[x['source_dir']]
        for g in x['groups']:
            values=[int(t) for t,n in g['duration_histogram_ns'].items() for _ in range(n)]
            assert len(values)==g['count'] and sum(values)==g['sum_ns']
            assert statistics.median(values)==g['median_ns']
    assert len(summary['families'])==8
    for f in summary['families']:
        assert len(f['per_rank'])==16 and f['rank_count']==16
        assert f['rank_median_duration_ns']==statistics.median(x['median_ns'] for x in f['per_rank'])
        assert f['count_per_rank_values']==([60] if f['label']=='markov_head_compatible' else [12])
    paths=[]
    for name in ('SCOPE-AND-GATES.md','COVERAGE-LEDGER.md','TAIL-HEAD-SAMPLER-REVIEW.md'):
        p=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not link.startswith(('https://','http://','#')):
                assert (p.parent/link.split('#')[0]).exists(),(p,link);paths.append(link)
    record.update(status='PASS_CHECKPOINT_ONLY',source_files=len(files),unique_source_sites=id_counts,
                  module_surfaces=26,whole_modules_closed=0,local_branch_contracts=9,
                  measured_vocab_families=8,csv_hashes_cross_checked=16,local_links_checked=len(paths),
                  overall_goal_complete=False)
except Exception:
    record.update(status='FAIL',error=traceback.format_exc())
(ROOT/'checkpoint-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
sys.exit(0 if record['status']=='PASS_CHECKPOINT_ONLY' else 1)
