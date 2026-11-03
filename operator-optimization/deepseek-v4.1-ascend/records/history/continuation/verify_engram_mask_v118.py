"""Independent demand-set, signed-zero and checkpoint verification, without target imports."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'engram-mask-inputs-v118.json')['inputs']:ck('input:'+r['path'],pinned(r))
    src=read(O/'engram-mask-source-contract-v118.json');texts={}
    for r in src['parts']:
        ck('source:'+r['path'],pinned(r));s=Path(r['path']).read_text();texts[r['path']]=s
        ck('clause:'+r['path'],s.splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    native=next(s for p,s in texts.items()if p.endswith('/triton/engram_gate.py'))
    common=next(s for p,s in texts.items()if p.endswith('/engram/common.py'))
    ck('late_native_mask_and_residual',native.index('gate = tl.where(active, gate, 0.0)')<native.index('h + gate * value')and native.index('rstd = tl.rsqrt')<native.index('active = tl.load(mask'))
    ck('no_FP_contraction_claim','enable_fp_fusion=False'in native)
    ck('empty_and_stride_guards','if tokens:'in native and 'raise ValueError'in native)
    ck('ordinary_late_mask','gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)'in common and '(hidden.float() + gate.unsqueeze(-1) * value.float()'in common)
    ck('current_native_hit_not_promoted',not src['native_unrotated']['actual_current_profile_hit'])
    profile=read(O/'engram-mask-profile-scope-v118.json')
    ck('five_shapes_208',len(profile['cohorts'])==5 and sum(r['executions']for r in profile['cohorts'])==208)
    ck('phase_unbound32_preserved',sum(r['executions']for r in profile['cohorts']if 'PHASE_UNBOUND'in r['context'])==32)
    ck('not_BOOL_field_identity',profile['unresolved_BOOL24_not_assigned_to_Engram_mask']and all(r['actual_mask_values_or_A']is None for r in profile['cohorts']))
    refs=read(O/'engram-mask-residual-reference-v118.json')
    ck('522240_scalar_references',refs['finite_BF16_hidden_by_eight_value_class_cases']==65280*8 and refs['unsafe_raw_copy_cases']==4)
    # Independent IEEE zero-sign/special-value witnesses, not target precision validation.
    def bits(x):return struct.unpack('>Q',struct.pack('>d',x))[0]
    ck('negative_zero_positive_value',bits(-0.0+0.0*1.0)==0 and bits(-0.0)==1<<63)
    ck('negative_zero_negative_value',bits(-0.0+0.0*-1.0)==1<<63)
    ck('positive_zero_negative_value',bits(0.0+0.0*-1.0)==0)
    ck('infinity_not_bypass',math.isnan(1.0+0.0*math.inf)and math.isnan(-1.0+0.0*-math.inf))
    ck('NaN_not_bypass',math.isnan(1.0+0.0*math.nan))
    models=read(O/'engram-ma<REDACTED_TOKEN>.json');ck('26_models',len(models['models'])==26)
    for row in models['models']:
        tt,aa,bb=row['T'],row['active_gate_rows'],row['certified_bypass_inactive_rows'];rr=tt-bb
        # Count independent output coordinates, then Dproducts for every selected coordinate.
        outputs=5120*(4*aa+rr)
        ck('case:'+str((tt,aa,bb)),row['value_rows']==rr and row['key_rows']==aa and row['demand_dense_MAC_reference']==6144*outputs and row['demand_BF16_KV_bank_bytes']==2*outputs)
        ck('ratio:'+str((tt,aa,bb)),Fraction(row['selected_GEMM_MAC_ratio'])==Fraction(outputs,tt*25600))
        keyblocks=set(range(4))if aa else set();valueblocks={4}if rr else set()
        ck('unique_weights:'+str((tt,aa,bb)),row['weight_unique_cold_payload_bytes']==6144*5120*2*len(keyblocks|valueblocks))
        ck('two_schedule_inputs:'+str((tt,aa,bb)),row['separate_value_then_active_key_input_interface_bytes']==12288*(rr+aa)and row['active_full_then_inactive_value_input_interface_bytes']==12288*rr)
        calls=([5]if aa else[])+([1]if rr>aa else[])
        ck('two_schedule_weights:'+str((tt,aa,bb)),row['active_full_then_inactive_value_weight_call_payload_bytes']==6144*5120*2*sum(calls))
        ck('all_rows_still_output:'+str((tt,aa,bb)),row['output_BF16_hidden_publication_bytes']==tt*40960)
    example=next(r for r in models['models']if (r['T'],r['active_gate_rows'],r['certified_bypass_inactive_rows'])==(3,1,0))
    ck('T3_A1_tradeoff',Fraction(example['selected_GEMM_MAC_ratio'])==Fraction(7,15)and example['active_full_then_inactive_value_weight_call_payload_bytes']==377487360 and example['weight_unique_cold_payload_bytes']==314572800)
    # Enumerate demand sets with small independent dimensions instead of repeating the big-shape formula.
    enumerated=0
    for tt in range(17):
        for aa in range(tt+1):
            for bb in range(tt-aa+1):
                for hc in range(1,5):
                    active=set(range(aa));bypassed=set(range(tt-bb,tt))if bb else set()
                    needed={(r,head,col)for r in range(tt)for head in range(hc+1)for col in range(3)
                        if (head<hc and r in active)or(head==hc and r not in bypassed)}
                    assert len(needed)==3*(hc*aa+tt-bb)
                    enumerated+=1
    ck('3876_independent_demand_sets',enumerated==3876)
    gate=read(O/'engram-ma<REDACTED_TOKEN>.json')
    ck('no_new_relations_or_native',all(gate[k]==0 for k in ['new_pairs','new_groups','new_native_promotions'])and not gate['whole_module_closed'])
    ck('no_numerical_physical_promotion',not models['actual_A_B_traffic_or_speedup_proved']and 'not a universal algorithmic lowerbound'in models['dense_domain'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v117.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v118.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        mat=read(O/'module-phase-branch-matrix-v118.json');ck('356_rows_26_modules',len(mat['modules'])==26 and sum(len(m['branches'])for m in mat['modules'])==356)
        m14=next(m for m in mat['modules']if m['id']=='M14');ck('M14_explicit_falsemask',len(m14['branches'])==21 and any(r['id']=='M14-OP-21'for r in m14['branches']))
        cov=read(O/'candidate-relation-coverage-v118.json');ck('134_26_461',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        old=read(O/'operator-analysis-coverage-ledger-v117.json');new=read(O/'operator-analysis-coverage-ledger-v118.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        obligations=read(O/'coverage-current-obligations-v118.json');gs={g['id']:g for g in obligations['obligations']}
        ck('text356_26_461','356 semantic'in gs['G00']['remaining']and '/26 groups leave461'in gs['G10']['remaining'])
        for r in read(O/'candidate-record-index-v118.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_active_loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v118.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,
        independent_small_demand_sets=enumerated,finite_host_scalar_references=522240,
        actual_mask_count_or_native_precision_attested=False,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'engram-mask-preflight-v118.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
