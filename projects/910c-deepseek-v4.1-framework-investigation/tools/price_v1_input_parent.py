#!/usr/bin/env python3
"""Price only matched existing parent scopes; no per-field invented costs."""
from pathlib import Path
from hashlib import sha256
from statistics import mean
import json
ROOT=Path('audit/execution-search')
def main():
    review=json.loads((ROOT/'v1-input-source-review.json').read_text());assert review['unresolved_input_or_control_sites']==86
    p=Path('audit/framework-timing-171/function-analysis.json');fine=json.loads(p.read_text())
    receipt_path=Path('audit/framework-timing-171/receipt.json');receipt=json.loads(receipt_path.read_text())
    config=json.loads((ROOT/'whole-forward-frozen-config.json').read_text());assert receipt['base_serve_sha256']==config['saved_serve']['sha256']
    oldp=ROOT/'common-device-price.json';old=json.loads(oldp.read_text())
    kinds=['AscendDSAV41CPMetadataBuilder.build','AscendDSAV41MetadataBuilder.build','AscendDSACPMetadataBuilder._build_local_token_metadata']
    ranks=[]
    for r in fine['ranks']:
        assert r['closed_cycles']==11
        rows=[next(x for x in r['functions'] if x['name'].endswith(':'+k)) for k in kinds]
        # These are disjoint function SELF partitions. They include retained work.
        gross=sum(x['prepare_self_free_ms'] for x in rows)
        match=next(x for x in old['ranks'] if x['ep']==r['ep']);assert abs(match['window_ms']-r['window_ms'])<1e-6
        ranks.append(dict(ep=r['ep'],window_ms=r['window_ms'],cycles=11,whole_three_function_self_Free_ms=gross,
            self_parts={k:x['prepare_self_free_ms'] for k,x in zip(kinds,rows)},prior_device_only_guarded_subset_ms=match['repeat_guarded_Free_ms'],light_metadata_frames=match['reference_light_metadata_frames']))
    window=mean(r['window_ms'] for r in ranks);gross=mean(r['whole_three_function_self_Free_ms'] for r in ranks)
    scenarios=[]
    ref=old['reference_light']['batch_seconds']
    for deletion_fraction in [0,.1,.25,.5]:
      for transfer in [0,.5,1]:
       for new_cost in [.1,.5,1]:
        values=[r['whole_three_function_self_Free_ms']/11*deletion_fraction*transfer-new_cost for r in ranks]
        scenario=dict(assumed_deletion_fraction=deletion_fraction,diagnostic_to_request_transfer=transfer,new_retirement_retention_and_other_ms_per_frame=new_cost,
            conditional_mean_net_ms_per_frame=mean(values),conditional_diagnostic_window_pct=mean(v*11 for v in values)/window*100,
            conditional_light_request_ms=mean(v*r['light_metadata_frames'] for v,r in zip(values,ranks)),conditional_light_request_pct=mean(v*r['light_metadata_frames'] for v,r in zip(values,ranks))/(ref*1000)*100)
        scenarios.append(scenario)
    result=dict(status='BOUNDED_MATCHED_WHOLE_PARENT_GROSS_AND_PAID_SENSITIVITY',matched_serve_sha256=receipt['base_serve_sha256'],load=receipt['load'],
        ranks=ranks,mean_window_ms=window,mean_three_function_self_Free_ms=gross,gross_diagnostic_window_pct=gross/window*100,
        gross_is_realizable_saving=False,previous_device_only_is_parent_upper_bound=False,
        scenarios=scenarios,default_unknowns=dict(assumed_deletion_fraction=0,diagnostic_to_request_transfer=0,new_cost_ms_per_frame='UNMEASURED'),
        reference_light_batch_seconds=ref,
        qualification='Disjoint CP/base/local-helper SELF exposure in the same16rank11cycle diagnostic, excluding separately listed marker boundary. Gross contains first producers, physical mapping, native packets and necessary retained work; NOT all deletable and NOT a native/request speedup. Fractions and transfer are explicit unknown sensitivities; cost pays new fences, source retention and other work. Previous helper/device proposals are subsets and not added. Light reference is mirrored balanced C8 only; original distinct-prompt C8 TTFT and six ordinary loads stay unknown.',
        decision='Larger frame is not pruned: material existing gross headroom, first-owner proof and real deletion/cost still determine its value. No additional fine profiling justified solely to price tiny QSL copies; establish whole transformation and reuse existing evidence first.',
        embedding_scope='Selected supports_mm_inputs config and final embedding implementation/profile events are not yet closed; no embedding-copy benefit or stop decision inferred from three-function parent.',
        inputs=[dict(path=str(x),sha256=sha256(x.read_bytes()).hexdigest()) for x in [ROOT/'v1-input-source-review.json',p,receipt_path,oldp]],
        new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'v1-input-parent-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    selected=next(s for s in scenarios if s['assumed_deletion_fraction']==.1 and s['diagnostic_to_request_transfer']==.5 and s['new_retirement_retention_and_other_ms_per_frame']==.1)
    print(json.dumps(dict(gross_Free_ms=gross,gross_window_pct=gross/window*100,selected_conditional_only=selected,new_NPU_work=0)))
if __name__=='__main__':main()
