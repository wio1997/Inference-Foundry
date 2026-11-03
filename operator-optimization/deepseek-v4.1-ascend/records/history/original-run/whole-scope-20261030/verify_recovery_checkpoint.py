#!/usr/bin/env python3
"""Check new recovery/contract references; no global or numerical acceptance."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback
import sys

W=Path(__file__).resolve().parent;R=W.parent
read=lambda p:json.loads(p.read_text())
record={'scope':'local checkpoint identities/references and explicit open-state invariants only','overall_goal_complete':False}
try:
    source=read(W/'coverage-ledger-v2/summary.json')
    assert source['source_files']==566 and source['parse_errors']==[] and not source['scope_closed'] and not source['overall_complete']
    files=read(W/'coverage-ledger-v2/source-files.json')
    branch={x['id']:x for x in read(W/'coverage-ledger-v2/branch-sites.json')}
    module=read(W/'module-coverage-ledger.json')
    assert not module['goal_scope_closed'] and not module['overall_complete']
    assert len(module['surfaces'])==26 and len(module['rule_coverage'])==260
    paths={x['path'] for x in files}
    for m in module['surfaces']:
        assert not m['complete'] and len(m['mandatory_gates'])==8
        assert all(x in paths for x in m['indexed_source_paths'])
        assert all(x in branch for x in m['indexed_branch_ids'])
        assert all(g['status'] not in ('PASS','CLOSED','COMPLETE') for g in m['mandatory_gates'].values())
    total=0
    for name,key in [('tail-branch-review-delta.json','branch_id'),('dspark-branch-review-delta.json','id')]:
        d=read(W/name)
        for row in d['reviews']:
            original=branch[row[key]]
            assert all(row[k]==original[k] for k in ['source','source_sha256','line','condition'])
            assert not row['whole_module_closed'] and not row['all_consumers_closed']
            total+=1
    assert total==23
    current=read(W/'STATE.json');outer=read(R/'STATE.json')
    assert current['goal_status']=='active' and not current['overall_complete'] and not current['source_scope_closed']
    assert not outer['overall_complete'] and outer['goal_tool_state_last_checked']=='active'
    assert current['whole_modules_closed']==0 and not current['target_operator_changes'] and not current['new_inference_or_npu_work_this_goal']
    task=W/'tasks/whole-opportunity-coverage'
    index=read(task/'index.json');assert index['status']=='ACTIVE' and index['completion_gate'] is None
    run=task/'loops/loop-001/runs/c8-input32k-output1k-prefix90/run-20261030T121953Z'
    result=read(run/'result.json');assert result['status']=='pass' and result['correctness']=='not-applicable' and result['metric'] is None
    assert read(task/'loops/loop-001/decision.json')['result']=='PENDING'
    links=0
    for name in ['PROGRESS.md','SCOPE-AND-GATES.md','COVERAGE-LEDGER.md','TAIL-HEAD-SAMPLER-REVIEW.md','DSPARK-CONTRACT-REVIEW.md']:
        p=W/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('https://','http://','#')):
                assert (p.parent/target.split('#')[0]).exists(),(name,target);links+=1
    catalog=read(R/'RESEARCH-CATALOG.json')
    checked=0
    for item in catalog['indexed_files']:
        if item.get('current_local_record'):
            data=Path(item['local_path']).read_bytes()
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['relative_path']
            checked+=1
    assert catalog['read_first'][0]=='whole-scope-20261030/STATE.json'
    assert 'DSpark引用的 `vllm.models.qwen3_dspark`' not in (W/'TAIL-HEAD-SAMPLER-REVIEW.md').read_text()
    validation=subprocess.run(['python3','/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py','validate','--task-dir',str(task)],capture_output=True,text=True,check=False)
    (W/'taskctl-validate-execution.json').write_text(json.dumps({'returncode':validation.returncode,'stdout':validation.stdout,'stderr':validation.stderr},ensure_ascii=False,indent=2)+'\n')
    assert validation.returncode==0,validation.stdout+validation.stderr
    reference=read(W/'references/pytorch-2.10-tensor-view.json')
    if reference['status']=='SNAPSHOT_SAVED':
        assert hashlib.sha256((W/'references/pytorch-2.10-tensor-view.html').read_bytes()).hexdigest()==reference['sha256']
    record.update(status='PASS_LOCAL_RECOVERY_CHECKPOINT_ONLY',whole_modules_closed=0,selected_local_contracts=total,
        local_links_checked=links,catalog_local_hashes_checked=checked,taskctl_validate_returncode=validation.returncode,reference_snapshot=reference['status'])
except Exception:
    record.update(status='FAIL',error=traceback.format_exc())
(W/'recovery-checkpoint-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
sys.exit(0 if record['status']=='PASS_LOCAL_RECOVERY_CHECKPOINT_ONLY' else 1)
