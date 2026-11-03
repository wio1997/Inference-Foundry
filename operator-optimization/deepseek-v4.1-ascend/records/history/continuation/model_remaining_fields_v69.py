"""Conditional operator alternatives and finite math witnesses, without execution claims."""
from collections import Counter
from itertools import product
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_state_engram_v62 import dimensions

def main():
    assert not (O/'remaining-field-fusion-costs-v69.json').exists()
    p=read(O/'remaining-field-regime-models-v69.json');g=read(O/'remaining-field-native-chains-v69.json')['chains'];bykey={tuple(k):r for r in p['regimes']for k in r['native_TASK_keys']}
    # Correct this uncheckpointed descriptive coefficient before it becomes authority.
    corrected=[]
    for r in p['regimes']:
        if r['operator']=='copy_and_expand_dflash_and_dspark_inputs_kernel':
            before=r['work_model']['query_source_value_reference_bytes'];assert before in ['(24+4*HAS_REJECTED)*B*Q','(28+4*HAS_REJECTED)*B*Q']
            r['work_model']['query_source_value_reference_bytes']='(28+4*HAS_REJECTED)*B*Q'
            corrected.append(dict(regime_id=r['regime_id'],before=before,after=r['work_model']['query_source_value_reference_bytes'],reason='QSLstart4+end4+GPUlength4+lastposition8+BT4+bonus4=28B/query; optionalreject4. Metadata reference loads, not HBM.'))
    assert len(corrected)==5
    next_costs=[]
    for x in g['next_ID_13TASK']:
        B,K=x['shape'];old=sum(bykey[(x['database_index'],s['TASK_rowid'])]['known_logical_value_bytes_excluding_control']for s in x['stages']);assert old==38*B*K+94*B
        next_costs.append(dict(database_index=x['database_index'],B=B,K=K,source_13TASK_known_value_bytes=old,fused_keep_row_values_and_read_all_backup_reference_bytes=4*B*K+16*B,logical_interface_budget_bytes=34*B*K+78*B,source_private_clone_additional_bytes=8*B*K,clone_SCOPE='SOURCE-only private copy; do not add to this13TASK count or M19 gains twice.',fused_plus_padded_inputs_reference_bytes=4*B*K+36*B-4 if K==6 else None,source_plus_padded_inputs_reference_bytes=38*B*K+122*B-4 if K==6 else None,rows_are_observed_BK_only_not_alias_generation_proof=True))
    rope=[]
    for name,n in [('cached',3),('advanced',2)]:
        l0,l1=(173,174)if n==3 else(185,186)
        a=g[f'rope_{l0}_{n}TASK'];b=g[f'rope_{l1}_{n}TASK'];assert len(a)==len(b)
        for x,y in zip(a,b):
            assert x['database_index']==y['database_index']
            rx=[bykey[(x['database_index'],s['TASK_rowid'])]for s in x['stages']];ry=[bykey[(y['database_index'],s['TASK_rowid'])]for s in y['stages']]
            get=lambda rs:next(r for r in rs if r['operator']in ['Index','GatherElementsV2'])
            N=dimensions(get(rx)['signature']['outputShapes_text'])[0][0];assert dimensions(get(ry)['signature']['outputShapes_text'])[0][0]==N
            old=sum(r['known_logical_value_bytes_excluding_control']for r in rx+ry);assert old==(4112 if name=='cached' else 1056)*N
            rope.append(dict(database_index=x['database_index'],kind=name,N=N,source_known_value_bytes=old,joint_row_lookup_reference_bytes=1032*N,logical_interface_budget_bytes=old-1032*N,row_cast_two_gathers_reference_bytes=1044*N if name=='cached'else None,required_two_FLOAT_outputs_bytes=512*N,pairing_scope='Source-order same-DB/shape only; config/group/position equality/alias not supplied by trace. Keep separate if not equal.',cast_policy='Native INT32 conversion and gather domain'if name=='cached'else 'INT64 negative-wrap and same bounds-check/error policy; cannot substitute nonnegative-only gather unconditionally.'))
    swa=[];slot=[]
    for r in p['regimes']:
        if r['family']=='SWA_PRODUCER':
            w=r['work_model'];B,N,C=w['B'],w['N'],w['C'];swa.append(dict(B=B,N=N,C=C,source_value_access_reference_bytes=r['known_logical_value_bytes_excluding_control'],same_layout_required_output_bytes=4*N*(C+1),unique_input_and_full_output_cold_model_bytes=8*B+4+4*N*(C+1),compact_required_output_bytes=4*B*(C+1),compact_output_budget_bytes=4*(N-B)*(C+1),saved_request_mapping_and_arithmetic='Per-request launch can avoid each query scanning all B request ends and recalculating window; full output writes remain unless consumer ABI changes.',condition='raw V retained, CP row mapping/masks/new planner/SMLA ABI/all readers verified; not contiguous N,C stride substitution.'))
        if r['family']=='GROUP_SLOT':
            w=r['work_model'];slot.append(dict(N=w['N'],B=w['B'],G=13,current_full_output_write_bytes='52*C',dirty_suffix_output_write_bytes=f'52*({w["N"]}+max(previous_N-{w["N"]},0))',steady_same_N_write_bytes=52*w['N'],shared_positions_reference_load_budget_bytes=8*w['N']*12,capacity_C_unbound=True,scope='C>=N; valid QSL partitions prefix. No group output is13logicalelements. Position sharing changes within-kernel tiling, not an additional group launch fusion.'))
    # Check exact count-index semantics, and produce two falsifying examples.
    cases=0
    def reference(row,V,backup,discard=False):
        a=[-1]*len(row)if discard else list(row);v=[x!=-1 and x<V for x in a];c=sum(v);return(a[max(c-1,0)]if c else backup,c)
    for K in range(1,7):
        for row in product([-2,-1,0,3,4],repeat=K):
            for discard in [False,True]:
                got=reference(row,4,99,discard);a=[-1]*K if discard else row;c=sum(1 for x in a if x!=-1 and x<4);fused=(a[c-1]if c else 99,c);assert got==fused;cases+=1
    assert reference([0,-1,3],4,99)==(-1,2)and reference([-2],4,99)==(-2,1)
    # Full-suffix invariant proof checked over all old/new prefixes, including shrink/grow.
    suffix_cases=0;mask_cases=0
    for C in range(1,25):
        for oldN in range(C+1):
            old=list(range(oldN))+[-1]*(C-oldN)
            for newN in range(C+1):
                new=old.copy();new[:newN]=list(range(100,100+newN));new[newN:oldN]=[-1]*max(oldN-newN,0)
                assert new==list(range(100,100+newN))+[-1]*(C-newN);suffix_cases+=1
                prior=[i<oldN for i in range(C)];expected=[i<newN for i in range(C)];lo,hi=sorted([oldN,newN]);prior[lo:hi]=[newN>oldN]*(hi-lo);assert prior==expected;mask_cases+=1
    # A stale bank and native block-number rounding falsify unconditional substitutions.
    stale=[5,6,7,-1];newN=1;bad=stale.copy();bad[:newN]=[100];assert bad!=[100,-1,-1,-1]
    import struct
    f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
    assert int(f32(16777217))==16777216
    variants=[]
    def v(i,choice,model,conditions,decision='CONDITIONAL_OPERATOR_OPTION'):
        variants.append(dict(id=i,choice=choice,cost_model=model,conditions=conditions,decision=decision,legal_for_all_captured_values_proved=False,native_speedup=None))
    v('FIELD-NEXT','Fuse13TASK predicate/count/count-index select','38BK+94B→4BK+16B; privateclone8BK SOURCE scope separate.','K1/6 exact FP32 count; literal negative predicate and holes/count-1; backup/discard, all readers, INT64 counts and INT32 IDs.')
    v('FIELD-NEXT-PREP','Fuse already dual-output padded-input formula with next-ID count','K6 possible72calls;38BK+122B-4→4BK+36B-4 with count still materialized.','cu/QSL and discard/backup ready, same rows/epoch; preserve d<=0 r0, no clamping; no matching tensor generation proven by time adjacency.')
    v('FIELD-NEXT-LAST','Replace count-1 by rightmost-valid index','Rejected: [0,-1,3] selects-1 at count-1, rightmost-valid would3.','Only additional prefix-validity proof permits different algorithm.','EXCLUDED_WITH_COUNTEREXAMPLE')
    v('FIELD-NEXT-NONNEG','Strengthen valid predicate to x>=0','Rejected: [-2] remains valid in baseline.','Vocabulary/range invariant not provided.','EXCLUDED_WITH_COUNTEREXAMPLE')
    v('FIELD-ROPE-CACHED','Reuse row-lookup alternative for cached cos/sin FLOAT banks','4112N→1044N rowcast+2gather or1032N joint; N10/20/24 observed.','INT32 conversion/error domain, table/group/config/stride equality, capture-fixed outputs and last readers; shared position/config only.')
    v('FIELD-ROPE-ADV','Joint uncached advanced row lookup/check','1056N→1032N including two baseline checks; output512N remains.','INT64 negative wrapping/same validation; check sideeffects and both table availability; no claim all table bytes read.')
    v('FIELD-ROPE-INLINE','Inline rowlookup into RoPE readers or share published cos/sin','Avoid private intermediate planes only once all readers accept new ABI.','Every consumer/config/bank epoch known; preserve FLOAT→consumer precision boundary and other C2/draft uses; do not add overlap with cached joint saving.')
    v('FIELD-MC2','Directly publish BOOL threshold mask','Current native INT32 broadcast/conversion intermediates→P BOOL stores.','A=num_actual_tokens;P=TP-padded tokens,0<=A<=P<=bankcap, all consumers see same bool bytes and generation.')
    v('FIELD-MC2-DIRTY','Maintain whole-bank threshold invariant across iterations','After full-bank establishment, abs(A-oldA) BOOL writes versusP; reset fallback fullP.','Track shared main/draft/capture bank identity/writers and readers; replay scalar binding must be valid; all C values established at initialization. No inference for ambiguous BOOL80.')
    v('FIELD-SP-NONE','Create SP false-prefix/true-pad mask in one producer','Remove standalone none-mask zeros only when no other reader; retain final mask.','Only is_padding=None; supplied true-prefix entries retained; SP rank split/order/capacity unchanged.')
    v('FIELD-SLOT-EXISTING','Use existing13group fusion/adaptive tiling as baseline','Already one launch; direct full output52*C bytes, not104B descriptor.','DCP1 fused guard/circular/group strides/window and FP32 lookup round-trip.','ALREADY_IMPLEMENTED_BASELINE')
    v('FIELD-SLOT-DIRTY','Clear only shrinking former active suffix','52*C→52*(N+max(oldN-N,0)) output writes, C unbound.','Established full PAD suffix for each actual group bank, previous writer end tracked, all writers confined prefix, readers done; dummy/group/capacity/alias/replay reset fallback.')
    v('FIELD-SLOT-POS','Share positions/QSL across group computation inside kernel','Positions reference8GN→8N;12*8N saved forG13, not actual HBM.','Group-specific divide/mod/table/circular retained; UB/register/occupancy/spill budget and tile dependencies; raw pointer arrays not logical outputs.')
    v('FIELD-WORKER-TAIL','Remove already-covered slot tail refill','Only same bank contained interval/PAD epoch/idempotence; table0 rowfills are separate.','7regimes936 TASKs not globally deletable; first/last readers and no intervening writer, matching full or dirty suffix proof.')
    v('FIELD-SWA-REQTILE','Generate current N,C indices using per-request tile','AvoidN scans of B QSL ends and repeat arithmetic; still4N(C+1) outputs.','Request sizes/partition and tile occupancy/capacity; current kernel already indices+rawlengths in one launch.')
    v('FIELD-SWA-COMPACT','Change indices/lengths to B request rows with query→request reuse','Full4N(C+1)→4B(C+1); Q5 observed,5x smaller output storage.','New SparseFlashMla/CP/planner consumer ABI, raw length notclipped, no overlapping writes/epoch, physical fallback/empty/captured branches. Cannot pass zero-stride expand as accepted ABI without proof.')
    v('FIELD-DRAFT-SHARED','Specialize per-group copy-expand to avoid repeated common fields','20T context per group plus(28+4R)BQ reads and12BQ+4BD query publications; group shared position/ID/indices can be once only ifGdraft>1.','ActualGdraft not inferred from targetG13. Group slots/tables remain separate; preserve accepted-context empty case/MM positions vslinearKV/DCP mapping; existing vectorcap/gridstride already baseline.')
    v('FIELD-DRAFT-LENGTH','Produce seed/length/QSL along existing draft-input producer','INT32→INT64 seed12B; seq Sub+Add20B with reject vsAdd8B; QSL arithmetic8(B+1), before constant operands.','Fixed seed tail0 still required; canonical GPU lengths/CPU mirrors, existing rejected consumer/all readers, new output ABI; not optimistic CPU lengths.')
    v('FIELD-DUMMY','Coalesce or reuse dummy initialization under exact writer epochs','Observed Fill stores are output extent; multiple banks can share launch, mandatory values persist.','real→dummy invalidates dirty assumptions;slot-1 vspositions127 vsindices0; all banks/reset/capture readers; constantindicesread-only only no mutation.')
    v('FIELD-OPAQUE-AICPU','Reclassify metadata planner resource without closing analysis','529TASKs allKERNEL_AICPU,blockNum0,4096B INT32 output; no Vector/Cube roofline assignment.','Need planner algorithm/input values/CPU or AICPU capacity/purity/address-binding/consumer ABI. No plan reuse orzero-work claim.','RESOURCE_TYPE_BOUND_INTERNAL_WORK_PENDING')
    write(O/'remaining-field-regime-models-v69.json',p)
    write(O/'remaining-field-fusion-costs-v69.json',dict(next_ID_models=next_costs,rope_lookup_models=rope,SWA_models=swa,group_slot_models=slot,reference_model='All costs logical value access/published bytes excluding stated controls; no measuredHBM/latency/criticalpath speedup. Typed integer/mod/index/Vector/AICPU bandwidth unbound. Standalone cold unique-input+materialized-output Q/B is conditional on actualHBM and no cache/overlap; not automatically physicallowerbound.',not_additive='Nested/alternative fusion and shared field budget counted once; saved ns cannot be summed into requestgain.',whole_modules_closed=0,overall_complete=False))
    write(O/'remaining-field-fusion-variants-v69.json',dict(variants=variants,variant_count=len(variants),whole_modules_closed=0,overall_complete=False))
    write(O/'remaining-field-math-validation-v69.json',dict(next_ID_exact_cases=cases,full_slot_suffix_cases=suffix_cases,whole_BOOL_threshold_cases=mask_cases,hole_counterexample=dict(row=[0,-1,3],V=4,count=2,baseline=-1,rightmost_valid=3),negative_counterexample=dict(row=[-2],V=4,baseline_valid=True),FP32_block_counterexample=dict(INT32=16777217,native_roundtrip=16777216),source_query_reference_correction=corrected,resolved_initial_import_error='Local extraction initially imported nonexistent S helper; fixed to W/frozen-source-snapshot before extraction/persistence. No authority mutation on failure.',profile_exact_cost_checks='88next13TASK;288cos+288sin cached;96cos+96sin advanced. All group costs checked against typed rows.',no_native_test=True,overall_complete=False))
    print(dict(next_groups=len(next_costs),rope_pairs=len(rope),SWA_models=len(swa),slot_models=len(slot),variants=len(variants),math_cases=[cases,suffix_cases,mask_cases]))

if __name__=='__main__':main()
