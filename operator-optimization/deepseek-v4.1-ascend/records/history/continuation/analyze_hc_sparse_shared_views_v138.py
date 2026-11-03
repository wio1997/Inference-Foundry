"""Positive CP shared-TopK view branch evidence for later attention consumers."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,json
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v138.json').exists()
    p=read(O/'hc-sparse-eager-native-frontiers-v138.json');groups=defaultdict(list);records=[]
    for r in p['records']:groups[r['database_index']].append(r)
    for ref in p['databases']:
        di=ref['database_index'];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};sids=[sid for sid,v in ss.items()if '/attention/context_parallel/dsa_v41_cp.py(206):'in v];assert sids
        fs=defaultdict(list)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE stack IN ('+','.join('?'for _ in sids)+')',sids):fs[f['id']].append(dict(f,value=ss[f['stack']]))
        ids=list(fs);aps=[dict(a,name_text=ss[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE callchainId IN ('+','.join('?'for _ in ids)+')',ids)if ss[a['name']]=='aten::slice']
        for r in groups[di]:
            parent=r['eager_HC_SP_parent'];ag,rs=parent['SP_allGather']['API'],parent['output_reduceScatter']['API'];hits=[a for a in aps if a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs'])]
            expect=r['index_mode']=='SHARED_TOPK';assert len(hits)==int(expect),(di,r['source_block_ordinal'],len(hits),expect)
            if expect:
                a=hits[0];records.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],source_topk_producer_block=r['source_ordinal_shared_topk_producer'],
                    shared_view_API=a,source_frames=fs[a['callchainId']],SMLA_TASK=r['stages']['SMLA']['TASK'],actual_bank_pointer_or_ID_values_or_read_completion_attested=False))
        db.close()
    assert len(records)==720
    groups_out=[]
    for src in p['records']:
        if not src['stages']['QLI']:continue
        cons=[r for r in records if r['database_index']==src['database_index']and r['forward_API_rowid']==src['forward_API_rowid']and r['source_topk_producer_block']==src['source_block_ordinal']]
        groups_out.append(dict(database_index=src['database_index'],forward_API_rowid=src['forward_API_rowid'],producer_block=src['source_block_ordinal'],QLI_TASK=src['stages']['QLI']['TASK'],PREPARE_TASK=src['stages']['PREPARE']['TASK'],
            source_SMLA_TASK=src['stages']['SMLA']['TASK'],later_source_positive_readers=[dict(block=r['source_block_ordinal'],shared_view_API_rowid=r['shared_view_API']['api_rowid'],SMLA_TASK_rowid=r['SMLA_TASK']['task_record_rowid'])for r in cons],
            source_role_lifetime_not_pointer_or_all_readers_closed=True))
    write(O/'hc-sparse-shared-topk-views-v138.json',dict(records=records,source_topk_groups=groups_out,source_positive_view_APIs=720,producer_groups=192,databases=p['databases'],
        producer_blocks=dict(Counter(g['producer_block']for g in groups_out)),actual_pointer_payload_epoch_or_native_fusion_proved=False,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='SHARED_TOPK_SOURCE_VIEWS_BOUND',view_APIs=720,producer_groups=192,new_device_work=False)))


if __name__=='__main__':main()
