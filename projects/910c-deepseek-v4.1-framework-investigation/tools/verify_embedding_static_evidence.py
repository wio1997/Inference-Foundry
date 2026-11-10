#!/usr/bin/env python3
"""Verify source identities, branch delta and old native endpoints; no new execution."""
from pathlib import Path
from hashlib import sha256
import json,zlib
ROOT=Path('audit/execution-search')
def read(n):return json.loads((ROOT/n).read_text())
def main():
    review=read('embedding-static-source-review.json');old=read('v1-input-source-review.json');price=read('first-static-parent-price.json');receipt=read('first-static-math-receipt.json')
    assert len(review['sources'])==len({x['runtime_path'] for x in review['sources']})==23
    for s in review['sources']:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    for s in review['prior_exact_SHA_references']:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['sha256']
    assert len(review['controls'])==29 and all(c['value']==c['expected'] for c in review['controls'])
    assert review['scoped_source_bindings']==77 and review['unresolved_input_or_control_sites']==79
    assert len(review['branches'])==len(old['branches'])==156
    changes=[(a,b) for a,b in zip(old['branches'],review['branches']) if a['condition_state']!=b['condition_state']];assert len(changes)==7
    assert all(a['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' and not b['fusion_search_complete'] and not b['observed_outcomes'] for a,b in changes)
    for n in ['embedding-entry-dependencies.json','embedding-dispatch-dependencies.json','embedding-registration-dependencies.json','embedding-method-dependencies.json','embedding-coordinate-dependencies.json','embedding-quant-description.json','first-static-parent-receipt.json','first-static-math-receipt.json','first-static-parent-price.json','embedding-static-source-review.json']:
        d=read(n)
        for pin in d.get('inputs',[]):assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256'],(n,pin['path'])
        assert not d.get('remote_writes') and not d.get('new_NPU_work') and not d.get('service_requests') and not d.get('new_profile_collection') and not d.get('new_profile')
    assert read('embedding-quant-description.json')['selected_description']['embed.weight']=='FLOAT'
    assert read('embedding-selected-launch.json')['sha256']==read('whole-forward-frozen-config.json')['saved_serve']['sha256']
    p=Path(receipt['raw_path']);assert sha256(p.read_bytes()).hexdigest()==receipt['raw_sha256'];raw=json.loads(zlib.decompress(p.read_bytes()));assert len(raw)==16
    assert sum(len(r['windows']) for r in raw)==3696
    active=[]
    for r in raw:
        for w in r['windows']:
            if not w['active']:continue
            active.append(w);assert w['math_boundary_qualified']
            for name in ['target','draft']:
                start=w['target_first_HcPre'][1] if name=='target' else w['first_HcPre_ns']
                end=w[name+'_pre_rope_end_task'];norm=w[name+'_cache_norm_task'];qb=w[name+'_Qb_task']
                assert norm[5]=='RmsNorm' and norm[6].endswith(',512;512"')
                if qb is not None:assert qb[5] in ['MatMulV2','QuantBatchMatmulV3'] and qb[7].endswith(',32768"')
                assert end==max([norm]+([qb] if qb else []),key=lambda t:t[2])
                assert start<end[2]<=w[name+'_prefix_end_task'][1] and w[name+'_static_after_HC_ns']==end[2]-start
            assert w['target_prefix_end_task'][1]<w['draft_start_task'][1]
    assert len(active)==receipt['qualified']==3624 and not receipt['unqualified']
    assert len(price['groups'])==453 and price['active_waves']==227
    assert sum(x['rank_windows'] for x in price['draft_lookup_population'])==3624
    assert '454' in price['physical_DP_qualification'] and not price['whole_chain_complete']
    assert not review['new_primitive_admissions'] and not review['new_framework_replacement_admissions'] and not review['whole_chain_complete']
    artifact_names=['embedding-static-source-review.json','embedding-static-source-excerpts.json','first-static-parent-price.json','first-static-math-receipt.json','embedding-static-frontier.json','词嵌入实际派发与首层静态计算整段判断.md']
    out=dict(status='PASS_BOUNDED_SOURCE_AND_EXISTING_TRACE_INTEGRITY_NOT_NATIVE_ADMISSION',current_sources=23,prior_HF_exact_SHA=1,original_method_controls=29,original_sites=156,source_constrained=77,input_control_pending=79,full_rank_windows=3696,active_rank_windows=3624,TP8_active_groups=453,physical_groups_in_active_waves=454,unpriced_synthetic_rank_windows=8,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in artifact_names],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'embedding-static-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','reader_sha256']}))
if __name__=='__main__':main()
