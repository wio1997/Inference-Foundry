"""Verify one output-cut delta; no native/model imports or old case replay."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import ast, json, math, struct, sys, textwrap
from finalize_origin_coverage_v9 import N, O, T as TASK_DIR, L as LOOP_DIR, read, write
from analyze_evidence_batches_v155 import pin


def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',float(value)))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return Fraction(struct.unpack('<f',struct.pack('<I',rounded))[0])


def main():
    preflight='--preflight'in sys.argv;checks=0
    def check(condition,label):
        nonlocal checks
        assert condition,label;checks+=1
    for row in read(O/'output-weight-plan-inputs-v157.json')['inputs']:check(pin(row['path'])==row,'input '+row['path'])
    for row in read(O/('checkpoint-manifest-v156.json'if preflight else'checkpoint-manifest-v157.json'))['artifacts']:check(pin(row['path'])==row,'current seal '+row['path'])
    source=read(O/'WOA-local-versus-full-source-contract-v157.json')
    for part in source['parts']:
        check(pin(part['source']['path'])==part['source'],'source snapshot pin')
        lines=Path(part['source']['path']).read_text().splitlines();check(lines[part['first_line']-1:part['last_line']]==part['lines'],'exact question slice')
    cp_call=next(p for p in source['parts']if p['question']=='Selected V4.1 CP restore and default projection call');tree=ast.parse(textwrap.dedent('\n'.join(cp_call['lines'])))
    calls=[node for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and node.func.attr=='_forward_o_proj'];check(len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords,'V4.1 call does not request full group mode')
    cp_def=next(p for p in source['parts']if p['question']=='Local-group default and WO_B call');tree=ast.parse(textwrap.dedent('\n'.join(cp_def['lines'])));function=tree.body[0]
    index=[x.arg for x in function.args.kwonlyargs].index('full_gather_wo_a_enabled');check(isinstance(function.args.kw_defaults[index],ast.Constant)and function.args.kw_defaults[index].value is False,'full group default false')
    group_mode=[node for node in ast.walk(tree)if isinstance(node,ast.IfExp)and isinstance(node.test,ast.Name)and node.test.id=='full_gather_wo_a_enabled'];check(len(group_mode)==1 and isinstance(group_mode[0].orelse,ast.Attribute)and group_mode[0].orelse.attr=='n_local_groups','local groups selected by declared default')
    generic='\n'.join(next(p for p in source['parts']if p['question']=='Generic all-gather fetches BOTH WO_A and WO_B')['lines']);check('self.wo_a_weight_method.all_gather_weight('in generic and'self.wo_b_weight_method.all_gather_weight('in generic,'generic mode gathers both weights')
    guard='\n'.join(next(p for p in source['parts']if p['question']=='Generic full-o-proj hardware/config guard')['lines']);check('enable_dsa_cp_full_o_proj() and is_950()'in guard,'generic hardware/config guard')
    helper='\n'.join(next(p for p in source['parts']if p['question']=='Active WO_A view/layout helper')['lines']);check('weight = self.wo_a.weight'in helper and'all_gather'not in helper,'helper is active-weight layout only')
    check(source['checkpoint_header_evidence_available']is False and source['all_rank_weight_values_or_physical_residency_closed']is False,'source evidence limits')
    parent=read(O/'hc-norm-attention-native-frontiers-v145.json');rows={(x['database_index'],x['forward_API_rowid'],x['source_block_ordinal']):x for x in parent['records']};native=read(O/'WOA-WOB-scoped-native-view-census-v157.json');seen=set()
    for row in native['records']:
        key=(row['database_index'],row['forward_API_rowid'],row['source_block_ordinal']);check(key not in seen,'unique reference instance');seen.add(key);base=rows[key]
        for name in['WO_A','WO_B']:
            check(row[name+'_TASK_rowid']==base['stages'][name]['TASK']['task_record_rowid']and row[name+'_COMPUTE_rowid']==base['stages'][name]['definition']['compute_record_rowid'],'reused exact native identity')
        check(row['WO_A_weight_view']==[1,4096,1024]and row['WO_B_weight_view']==[5120,1024]and row['dtype']=='BF16','local operator view shapes')
        check(math.prod(row['WO_A_weight_view'])*2==row['WO_A_logical_view_bytes']==8*2**20 and math.prod(row['WO_B_weight_view'])*2==row['WO_B_logical_view_bytes']==10*2**20,'view content sizes')
        check(base['collectives']['OUTPUT_RS']['COMM']['count']==row['M']*5120==row['RS_reported_count']and row['G']==8*row['M'],'recorded RS domain')
        check(not row['whole_storage_or_state_identity_proved'],'view does not identify whole storage')
    check(len(seen)==len(rows)==native['record_count']==960,'all selected reference frontiers covered')
    plans=read(O/'output-chain-weight-and-delivery-plans-v157.json');check(len(plans['models'])==20,'four actual geometries and five reuse epochs')
    for model in plans['models']:
        M=model['M'];R=model['same_layer_same_weight_calls'];S=M*R;check(S==model['sum_M'],'uniform reuse extent')
        # Independent source->recipient accounting in the declared protocol.
        A=B=C=0
        for src in range(8):
            for dst in range(8):
                if src==dst:continue
                A+=(4096+5120)*2*S
                B+=(1024+5120)*2*S+4096*1024*2
                C+=(4096*1024+5120*1024)*2
        A//=8;B//=8;C//=8
        check((A,B,C)==tuple(model[x]['total_delivery_bytes']for x in['A','B','C']),'complete three-graph delivery accounting')
        check(Fraction(B,A)==Fraction(2,3)+Fraction(4096,9*S)and Fraction(C,A)==Fraction(1024,S),'closed complete-cut ratios')
        check((A>B)==(3*S>4096)and(A>C)==(S>1024)and(B>C)==(3*S>2560),'all three break-even domains')
        for name,num,den in [('B_to_A',B,A),('C_to_A',C,A),('C_to_B',C,B)]:check(Fraction(model[name]['numerator'],model[name]['denominator'])==Fraction(num,den),'exact ratio serialization')
        check(model['wire_HBM_or_latency_gain_proved']is False,'no physical/time inference')
    for case in plans['logical_content_budget_boundary_cases']:
        S=case['sum_M'];cost={'A':7*18432*S,'B':7*(12288*S+8388608),'C':7*18874368};cap=case['declared_extra_distinct_weight_content_budget'];eligible=[x for x,n in [('A',0),('B',56*2**20),('C',126*2**20)]if cap>=n]
        check(eligible==case['eligible']and cost==case['logical_delivery_costs'],'budget domain/cost')
        best=min(cost[x]for x in eligible);check(case['best_delivery_variants']==[x for x in eligible if cost[x]==best],'content-budget optimum and ties')
        check(case['is_actual_allocation_or_performance_policy']is False,'budget is logical content only')
    check(3*1365<4096<3*1366 and 3*853<2560<3*854,'strict integer threshold boundaries')
    witness=read(O/'output-WO_B-partial-round-witness-v157.json');partial=[]
    for group in witness['input_and_weight_groups']:
        x=list(map(Fraction,group['input']));weight=list(map(Fraction,group['weight']));check(all(bf16(v)==v for v in x+weight),'BF16 input values exact');partial.append(sum(a*b for a,b in zip(x,weight)))
    original=bf16(sum(bf16(v)for v in partial));full=bf16(sum(partial));replay=bf16(sum(bf16(v)for v in partial))
    check([str(x)for x in partial]==witness['unrounded_partial_sums'],'exact partial dot products')
    check(original==replay==Fraction(1)and full==Fraction(129,128)and original!=full,'independent BF16 bit-round counterexample')
    check(witness['actual_loaded_CANN_numeric_failure_proved']is False and witness['does_not_reject_faithful_group_partial_round_replay'],'counterexample scope')
    route=read(O/'WOA-evidence-route-status-v157.json');check(not route['new_checkpoint_header_or_loader_source_evidence_acquired']and not route['fresh_identity_verified'],'failed remote attempt not evidence')
    check(read(N/'rank157-identity-recheck.json')['stdout']=='match asset failed: Core API 发生错误'and read(N/'original166-xinyufeng-recheck-v157.json')['returncode']==255,'actual captured connection results')
    check(not(O/'WOA-saved-artifacts-v157.json').exists()and not(N/'WOA-artifact-cache-v157').exists(),'no fabricated checkpoint cache')
    summary=read(O/'output-weight-plan-summary-v157.json');check(summary['new_pair_clauses']==summary['new_independence_proofs']==summary['new_native_operand_or_phase_promotions']==summary['new_whole_module_closures']==0 and summary['overall_complete']is False,'no unsupported completion')
    if not preflight:
        cp=read(O/'REVIEW-CHECKPOINT-v157.json');check(cp['summary']==read(N/'RECOVERY.json')['latest_profile_first_review']['summary'],'recovery authority')
        check(cp['candidate_ledger'].endswith('candidate-profile-ledger-v154.json')and cp['unchanged_operator_authorities_reused'],'unchanged operator authority')
        check(read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(LOOP_DIR/'state.json')['status']=='EVALUATING','Task/Loop statuses')
        current=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');check(current['effective_checkpoint']==157 and pin(current['work_queue']['path'])==current['work_queue'],'workflow pointer')
    result=dict(status='PASS',checkpoint=157,preflight=preflight,checks=checks,verification_scope=['Exact operator-source slices/default and guarded full-weight call differences','960 reused native WO_A/WO_B/RS reference frontiers without old identity-suite rerun','Complete output communication delivery models with distinct weight contents and break-even domains','Independent BF16 bit-round witness;failed remote collection remains missing evidence'],old_unchanged_operator_case_suites_reexecuted=False,new_device_work=False,new_native_or_whole_closures=0,overall_complete=False,observed_utc=datetime.now(timezone.utc).isoformat(),verifier=pin(Path(__file__)))
    output=O/('output-weight-plan-preflight-v157.json'if preflight else'output-weight-plan-postcheck-result-v157p.json');check(not output.exists(),'single result receipt')
    if not preflight:
        manifest=O/'checkpoint-manifest-v157.json';result.update(sealed_manifest=pin(manifest),sealed_manifest_files=len(read(manifest)['artifacts']),recovery_summary_agrees=True,sealed_files_modified=False)
    write(output,result)
    if not preflight:write(N/'CURRENT-POSTCHECK.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
