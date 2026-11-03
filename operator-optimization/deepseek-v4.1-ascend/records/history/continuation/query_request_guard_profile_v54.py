"""Immutable32DB metadata plus recorded request-processor stack census."""
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import json
import sqlite3
from finalize_origin_coverage_v9 import N, O, read, write
from mirror_saved_profile_core_v53 import ORIGINAL, CACHE


def main():
    assert not (O/'request-guard-saved-profile-v54.json').exists()
    caps=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')
    allow={'ASCEND_GLOBAL_LOG_LEVEL','HCCL_RDMA_TC','HCCL_RDMA_SL','ACLNN_CACHE_LIMIT',
        'HOST_CACHE_CAPACITY','ASCEND_ENHANCE_ENABLE','PYTORCH_NPU_ALLOC_CONF','ASCEND_LAUNCH_BLOCKING','HCCL_ALGO'}
    terms=['watermark','logits_processor','repetition','min_p','logprob','penalties','topk_topp','top_k_top_p']
    rows=[];frames=Counter()
    for i,r in enumerate(caps['databases']):
        p=CACHE/Path(r['path']).relative_to(ORIGINAL);before=p.stat();assert before.st_size==r['bytes']
        db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('pragma query_only=on')
        metas=[]
        for rid,name,val in db.execute('select rowid,name,value from META_DATA'):
            if name not in ['ENV_VARIABLES','parallel_group_info']:continue
            obj=json.loads(val)
            if name=='ENV_VARIABLES':obj={k:v for k,v in obj.items()if k in allow}
            metas.append(dict(rowid=rid,name=name,value=obj))
        ranks=db.execute('select rankId,deviceId from RANK_DEVICE_MAP').fetchall()
        sql='select id,value from STRING_IDS where '+' or '.join('lower(value) like ?'for _ in terms)
        strs=db.execute(sql,tuple('%'+s+'%'for s in terms)).fetchall();hits=[]
        for sid,value in strs:
            # Definitions/stack incidences are not invocation counts.
            count=db.execute('select count(*)from PYTORCH_CALLCHAINS where stack=?',(sid,)).fetchone()[0]
            apis=db.execute('select count(*)from PYTORCH_API where name=?',(sid,)).fetchone()[0]
            if count or apis:
                hits.append(dict(string_id=sid,value=value,callchain_frame_rows=count,API_name_rows=apis))
                frames[value]+=count
        db.close();after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        rows.append(dict(database_index=i,sha256=r['sha256'],rank_device_rows=ranks,metadata=metas,processor_string_hits=hits,source_stat_unchanged=True))
    write(O/'request-guard-saved-profile-v54.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),databases=rows,
        keyword_terms=terms,all32_databases_read=True,recorded_frame_totals=dict(frames),
        count_scope='Stack-frame rows count incidences in saved callchain definitions,not function invocations or CPU time. Missing frames do not exclude compiled or unrecorded paths.',
        duplicate_META_DATA_names_preserved=True,no_new_source_model_device_remote_work=True,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(databases=len(rows),unique_recorded_processor_frames=len(frames),
        watermark_frames=[f for f in frames if 'watermark'in f.lower()],overall_complete=False)))


if __name__=='__main__':main()
