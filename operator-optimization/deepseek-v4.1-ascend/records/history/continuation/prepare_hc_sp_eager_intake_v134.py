"""Bound the remaining eager COMM population without inventing a TASK join."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    inputs=[O/'hc-sp-native-frontiers-v133.json',O/'communication-native-operator-population-v66.json']
    used={(r['database_index'],r['SP_allGather']['COMM_OP_rowid'])for r in read(inputs[0])['records']}
    rest=[r for r in read(inputs[1])['operations']if r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'and(r['database_index'],r['COMM_OP_rowid'])not in used]
    assert len(rest)==960 and all(r['TASK_rowid']is None and r['source_grade']=='DIRECT_NATIVE_FULLCHAIN'and r.get('graph_ordinal')is None for r in rest)
    counts=Counter((r['count'],r['rankSize'],r['dtype'])for r in rest)
    assert counts=={(4096000,8,'BFP16'):320,(4177920,8,'BFP16'):320,(4270080,8,'BFP16'):320}
    out=dict(status='NEXT_EAGER_COMM_POPULATION_BOUNDED_NOT_HC_JOINED',inputs=[pin(p)for p in inputs],remaining_operators=rest,remaining_count=960,
        regimes=[dict(count=k[0],rankSize=k[1],dtype=k[2],executions=v,sender_rows_from_count=k[0]//5120,uniform_gather_padded_rows=k[1]*k[0]//5120)for k,v in counts.items()],
        all_COMM_without_unique_COMPUTE_TASK=True,
        next_question='BindHC/RMS/Postandpremix/Engram/auxbyexactCID/sourcecaller/operatorfrontiers. TheseCOMMrecordsremainnativecommunicationevidencewithoutauniqueCOMPUTE_TASKnode;neverrequireorinventthatTASK. Paddedrowsdonotprovevalidpositions/actualN.',
        no_rank_episode_native_operand_or_new_relation_promotion=True,new_device_work=False,overall_complete=False)
    write(O/'hc-sp-eager-intake-v134.json',out)
    print(json.dumps(dict(status=out['status'],remaining=960,all_TASK_none=True,regimes=out['regimes'],new_device_work=False)))


if __name__=='__main__':main()
