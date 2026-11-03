"""Profile-led mixed BF16/FLOAT sibling projections and source-graph reconciliation pins."""
from pathlib import Path
from collections import defaultdict
import hashlib,zipfile
from finalize_origin_coverage_v9 import O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def receipt(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    assert not(O/'mixed-SWA-C2-native-frontiers-v88.json').exists()
    archive=read(O/'DSpark-tail-source-index-v85.json')['original_archive'];files=[]
    clauses={
      'vllm-ascend/vllm_ascend/attention/dsa_v41.py':[(350,385),(440,495),(497,549)],
      'vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py':[(15,25)],
      'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py':[(1510,1534),(1608,1617),(2205,2226)],
      'vllm-ascend/vllm_ascend/spec_decode/utils.py':[(440,498),(500,521),(523,565),(567,633)],
      'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py':[(433,441)]}
    with zipfile.ZipFile(archive['path'])as z:
        for member,ranges in clauses.items():
            p=W/'frozen-source-snapshot'/member;data=p.read_bytes();assert data==z.read(member)
            lines=data.decode().splitlines();assert all(b<=len(lines)for a,b in ranges)
            files.append(dict(**receipt(p),archive_member=member,clauses=[dict(first=a,last=b,source=lines[a-1:b])for a,b in ranges]))
    primary=read(O/'A3-datapath-primary-evidence-v80.json');prior=['projection-operator-numerical-contract-v61.json',
      'A3-datapath-primary-evidence-v80.json','joint-operator-resource-cut-proof-v83.json','M20-Markov-confidence-and-verification-contract-v36.json']
    write(O/'mixed-SWA-C2-source-contract-v88.json',dict(source_files=files,original_archive=archive,
      prior_input_receipts=[receipt(O/n)for n in prior],saved_primary_sources=primary['primary_sources'],
      facts=['SWA BF16 projection and C2hidden.float()+twoFLOAT projections consume the same source cache_hidden in the declared normal C2-source path;native actual pointers are not attested.',
        'C2 input widening already occurs once for both FLOAT projections;this is not a new cast-CSE opportunity.',
        'C2 wkv/wgate parameters declared FLOAT,whereas ratio1compressor wkv isBF16. FLOAT/HF32 active instruction mode and actual BF16 representability of weights are unknown.',
        'A3arch2201 separate AIC/AIV usesGMforCV;the newer UB→L1 direct route is not granted. L1only ingress is one domain;documentedGM→L0 alternatives prevent a universalL1bound.',
        'Confidence raw input/ordered seed sequence/global survival policy were settled atv36;this turn synchronizes stale graph edges without recounting source discovery.'],
      new_target_source_files_obtained=0,new_remote_source_read=False,new_public_document_read=False,
      current_CANN_9p2_loaded_all_paths_or_native_alias_proved=False,overall_complete=False))
    a_path=CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json'
    grouped=defaultdict(lambda:defaultdict(list))
    for a in read(a_path)['execution_assignments']:
        if a[2]=='GRAPH'and a[9]in[2,8,14]:grouped[a[0]][(a[8],a[9])].append(a)
    results=[];inputs=[receipt(a_path)]
    old={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'state-engram-graph-operator-chains-v62.json')['C2_chains']}
    for db,groups in grouped.items():
        p=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';data=read(p);inputs.append(receipt(p))
        tasks={t['task_record_rowid']:t for t in data['TASK_records']};defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']}
        for key,ass in groups.items():
            swa=[];comp=[];cast=[]
            for a in ass:
                t=tasks[a[1]];d=defs[t['globalTaskId']];shape=dimensions(d['inputShapes_text'])
                row=dict(TASK_rowid=a[1],globalTaskId=t['globalTaskId'],operator=d['opType_text'],
                  input_shapes=d['inputShapes_text'],output_shapes=d['outputShapes_text'],input_types=d['inputDataTypes_text'],output_types=d['outputDataTypes_text'])
                if d['opType_text']in['MatMulV2','MatMulV3']and shape==[[24,5120],[512,5120]]:
                    if d['inputDataTypes_text']=='DT_BF16;DT_BF16':swa.append(row)
                    elif d['inputDataTypes_text']=='FLOAT;FLOAT':comp.append(row)
                if d['opType_text']=='Cast'and shape==[[24,5120]]and d['inputDataTypes_text']=='DT_BF16'and d['outputDataTypes_text']=='FLOAT':cast.append(row)
            assert(len(swa),len(comp),len(cast))==(1,2,1)
            prior=old[(db,*key)]
            results.append(dict(database_index=db,graph_ordinal=key[0],layer=key[1],root_bound=prior['root_bound'],N=24,H=5120,D=512,
              SWA_BF16=swa[0],C2_FLOAT_projections=comp,C2_input_cast=cast[0],C2_pool_TASK=prior['pool']['TASK_rowid'],
              actual_same_storage_address_or_loaded_FLOAT_HF32_mode_or_weight_values_attested=False))
    assert len(results)==240 and sum(r['root_bound']for r in results)==192
    write(O/'mixed-SWA-C2-native-frontiers-v88.json',dict(frontiers=results,count=240,root_bound=192,phase_unbound=48,
      distinct_graphs=80,layers=[2,8,14],reused_matrix_executions=720,reused_input_cast_executions=240,input_receipts=inputs,
      source_value_contract_not_pointer_alias_or_critical_path_proof=True,new_regimes_or_executions_promoted=0,overall_complete=False))
    print(dict(source_files=5,mixed_frontiers=[240,192,48],reused_GEMMs=720,reused_casts=240,new_native_work=False))

if __name__=='__main__':main()
