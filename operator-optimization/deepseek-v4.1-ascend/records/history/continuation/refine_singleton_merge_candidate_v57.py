"""Separate a singleton Add specialization from full FFN fusion and from measured gain."""
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write


def main():
    assert not(O/'F23-singleton-merge-specialization-v57.json').exists()
    ledger=read(O/'operator-shape-ceiling-ledger-v55.json');refs=[]
    for r in ledger['observed_regimes']:
        s=r['signature']
        if r['operator']=='Add'and s['inputDataTypes_text']=='DT_BF16;DT_BF16':
            refs.append(dict(regime_id=r['regime_id'],modules=r['module_candidates'],signature=s,origin=r['origin'],context=r['context'],executions=r['executions'],timing=r['measured_native_task_duration']))
    assert refs
    prof=read(O/'shared-expert-shape-ceiling-v57.json')
    write(O/'F23-singleton-merge-specialization-v57.json',dict(
        source_contract='merge-to-HC-contract-v22.json:torch._foreach_add([shared],[routed])[0],alpha defaults1,Tensorresult.',
        transformation='Specialize listlength1 to a standard BF16 Add with identicaloperands,alpha1,broadcast/stride/output ownership;or fuse intoDown whilepreservingBF16round.',
        root_bound_ForeachAddList_time_ns=prof['root_bound_graph_stage_times_ns']['MERGE'],
        present_profile_Add_references=refs,
        same_shape_BF16_Add_reference_available=any(r['signature']['inputShapes_text']=='"3,5120;3,5120"'for r in refs),
        measured_replacement_speedup=None,
        logical_M3_work=dict(elements=15360,adds=15360,read_two_write_one_BF16_bytes=92160,
                            cold_reference_1p6TB_transfer_us=0.0576),
        conclusion='Singletonoperator replacement is a simplerconditionalalternative tofull FFN fusion. Currentforeachmedian20.561us meritspriority;largerBF16Add references are3.88..4.35us,butdifferentcontexts/shapes cannot establishreplacementgain.',
        conditions=['Samealpha1,tensorpair,dtype,shape/broadcast/layout/stride,NaN/Inf/rounding andalias/output ownership.',
            'No observablelistconsumer;currentsource extractssoleTensor. Preserveallrouted/sharedscaling/reduction/posthooks.',
            'No newoperator/nativebenchmark performed. Reference difference isnot a measured17us saving orendtoendgain.'],
        conditional_semantic_analysis=True,implementation=False,native_test=False,overall_complete=False))
    p=O/'shared-expert-fusion-variants-and-interactions-v57.json';d=read(p)
    d['variants'].append(dict(id='F23-ONEADD',proposal='Specialize singleton ForeachAddList to normal Add',condition='Preserveexistingalpha1/Tensorsemantics/alllayouts/numerics;profilecrossshapereferenceispriorityevidence,not measuredgain.'))
    d['interactions'].append(dict(a='F23-ONEADD',b='F23-DOWNADD',relation='ALTERNATIVE_WAYS_TO_REMOVE_GENERIC_MERGE;DO_NOT_ADD_BOTH_GAINS'))
    write(p,d)
    print('Singletonmergecandidateanalyzed;7variants;profilegainnotclaimed.')


if __name__=='__main__':main()
