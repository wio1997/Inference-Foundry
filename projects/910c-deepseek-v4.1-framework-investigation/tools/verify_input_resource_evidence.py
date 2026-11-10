#!/usr/bin/env python3
"""Check branch/domain/native-window evidence without re-running the model."""
from pathlib import Path
from hashlib import sha256
import json,zlib
ROOT=Path('audit/execution-search')
def read(n):return json.loads((ROOT/n).read_text())
def main():
    r=read('input-resource-source-review.json');old=read('embedding-static-source-review.json');plan=read('input-resource-source-plan.json');inv=read('input-resource-callsite-inventory.json');rawreceipt=read('input-kernel-windows-receipt.json');price=read('input-kernel-island-price.json')
    assert len(r['sources'])==len({s['runtime_path'] for s in r['sources']})==30
    current_inventory={s['runtime_path']:s['sha256'] for s in inv['inventory']};assert len(current_inventory)==inv['python_files']==756
    for s in r['sources']:
        assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
        if '/vllm-ascend/vllm_ascend/' in s['runtime_path']:assert current_inventory[s['runtime_path']]==s['runtime_sha256']
    assert len(r['branches'])==len(old['branches'])==156 and r['scoped_source_bindings']==90 and r['unresolved_input_or_control_sites']==66
    changes=[(a,b) for a,b in zip(old['branches'],r['branches']) if a['condition_state']!=b['condition_state']];assert len(changes)==13
    assert all(a['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' and not b['fusion_search_complete'] and not b['observed_outcomes'] for a,b in changes)
    assert plan['resource_specs']==54 and plan['framework_KV_groups']==13 and plan['runner_attention_backend_spec_groups']==26 and plan['draft_attention_groups']==1 and plan['draft_layer_group_ids']==[12,12,12]
    assert sum(x['layer_count'] for x in plan['groups'])==54 and sum(x['distinct_selected_backend_spec_pairs'] for x in plan['groups'])==26 and plan['counterfactual_one_padded_page_changes_attention_group_count']==2 and plan['source_specs_not_mutated']
    assert read('input-retirement-controls.json')['count']==19
    flags=read('input-frozen-flags.json');assert 'multistream_engram_overlap' in flags['saved_launch']['selected_options']['--additional-config']['absent_selected_suboptions'] and r['selected_A3_engram_overlap'] is False
    for n in ['input-resource-source-plan.json','input-retirement-controls.json','input-resource-source-review.json','input-kernel-island-price.json']:
        d=read(n)
        for pin in d.get('inputs',[]):assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256'],(n,pin['path'])
    p=Path(rawreceipt['raw_path']);assert sha256(p.read_bytes()).hexdigest()==rawreceipt['raw_sha256'];raw=json.loads(zlib.decompress(p.read_bytes()));assert len(raw)==16 and sum(len(x['windows']) for x in raw)==3696
    active=[w for rank in raw for w in rank['windows'] if w['active']];assert len(active)==3624 and not rawreceipt['invalid']
    for w in active:
        assert len(w['expansion'])==1 and len(w['engram'])==3 and all(v==1 for v in w['engram_counts'].values())
        assert max(t[2] for t in w['engram'])<=w['expansion'][0][1] and len({t[3] for t in w['engram']})==1
    assert len(price['groups'])==453 and price['active_waves']==227
    assert price['gross']['joint_named_islands']['TTFT_P90']['signed_percent']<1 and price['gross']['joint_named_islands']['makespan']['signed_percent']<1
    assert 'excludes' in price['scope'] and not r['whole_chain_complete'] and not r['new_primitive_admissions'] and not r['new_framework_replacement_admissions']
    for n in ['input-resource-dependencies.json','input-group-dispatch-dependencies.json','input-resource-callsite-inventory.json','input-frozen-flags.json','input-retirement-controls.json','input-kernel-windows-receipt.json','input-kernel-island-price.json','input-resource-source-review.json']:
        d=read(n);assert not d.get('remote_writes') and not d.get('new_NPU_work') and not d.get('new_profile') and not d.get('service_requests')
    artifacts=['input-resource-source-review.json','input-resource-source-plan.json','input-retirement-controls.json','input-kernel-windows-receipt.json','input-kernel-island-price.json','input-frozen-flags.json','input-resource-frontier.json','实际资源组与Engram_DSpark坐标及退役判断.md']
    out=dict(status='PASS_SCOPED_SOURCE_RESOURCE_NATIVE_WINDOW_INTEGRITY_NOT_PRIMITIVE_ADMISSION',current_source_pins=30,original_sites=156,scoped_bindings=90,pending_original_sites=66,resource_specs=54,KV_groups=13,runner_builder_groups=26,draft_builder_groups=1,original_Python_retirement_controls=19,coordinate_counterexamples=2,native_full_rank_windows=3696,native_active_rank_windows=3624,one_expansion_and_three_hash_tasks_each_active_window=True,lexical_inventory_files=756,lexical_inventory_not_full_source_review=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in artifacts],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'input-resource-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','reader_sha256']}))
if __name__=='__main__':main()
