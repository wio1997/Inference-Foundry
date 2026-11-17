"""Check actual target/draft receipts from all workers; no inference."""
import json
from pathlib import Path
def validate(directory,plan):
    d=Path(directory);p=json.loads(Path(plan).read_text());pids=set();ep=set();checks=total=0
    assert {x.name for x in d.glob('mapping_audit_rank*.json')}=={'mapping_audit_rank%d.json'%r for r in range(16)}
    assert {x.name for x in d.glob('mapping_draft_audit_rank*.json')}=={'mapping_draft_audit_rank%d.json'%r for r in range(16)}
    for r in range(16):
        t=json.loads((d/('mapping_audit_rank%d.json'%r)).read_text());q=json.loads((d/('mapping_draft_audit_rank%d.json'%r)).read_text())
        assert t['status']=='ACTUAL_PREPROCESS_LOAD_CONTENT_AND_DYNAMIC_GRAPH_LOOKUP_PASS'
        assert q['status']=='ACTUAL_DRAFT_LOAD_MAPPING_EXCLUSION_PASS'
        assert t['global_rank']==q['global_rank']==r and t['ep_rank']==q['ep_rank']
        assert t['pid']==q['pid'] and t['mapping_sha256']==q['mapping_sha256']==p['mapping_sha256']
        assert t['source_sha256']==q['source_sha256']==p['source_sha256']
        assert t['ep_size']==16 and t['moe_tp_size']==1 and t['target_layers']==40 and t['routed_slots_per_rank']==25
        assert t['allcontent_eq_count']==9000 and t['coverage_per_layer']==225
        assert t['dynamic_lookup_layers']==40 and t['changed_input_graph_layers']==list(range(40))
        assert q['drafts'] and all(x['loads']>0 and not x['uses_expert_location'] for x in q['drafts'])
        checks+=t['allcontent_eq_count'];total+=t['allcontent_bytes_compared'];pids.add(t['pid']);ep.add(t['ep_rank'])
    assert len(pids)==16 and ep==set(range(16)) and checks==144000
    return {'status':'ALL16_ACTUAL_CONTENT_DYNAMIC_GRAPH_DRAFT_EXCLUSION_PASS','container_worker_pids':sorted(pids),'tensor_shard_checks':checks,'bytes_compared':total,'mapping_sha256':p['mapping_sha256'],'scope':'Actual all-content prepostprocess weight/scale/offset copies and native synthetic all-ID eager/changed-input lookup; full-model output quality and TPS remain unverified.'}
