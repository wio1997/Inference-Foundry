"""Bind the documented host-argument API to each original eager MemSet launch."""
from collections import Counter,defaultdict
from bisect import bisect_left,bisect_right
from pathlib import Path
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v126.json').exists()
    old=read(O/'operand-correlation-native-records-v125.json')
    bydb=defaultdict(list)
    for r in old['records']:
        if r['operator']=='MemSet':bydb[r['database_index']].append(r)
    rows=[];dbrefs=[];counts=Counter()
    for db in old['databases']:
        p=Path(db['path']);identity=pin(p);assert identity['sha256']==db['sha256']
        before=p.stat();con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
        names={r['id']:r['value']for r in con.execute("SELECT *FROM STRING_IDS WHERE value LIKE 'aclrtLaunch%' OR value='launch'")}
        tids=defaultdict(list)
        for name_id,name in names.items():
            if name!='aclrtLaunchKernelWithHostArgs':continue
            for r in con.execute('SELECT rowid AS cann_api_rowid,*FROM CANN_API WHERE name=?',(name_id,)):
                tids[r['globalTid']].append(dict(r,name_text=name))
        for tid in tids:tids[tid].sort(key=lambda r:r['startNs'])
        starts={tid:[r['startNs']for r in rr]for tid,rr in tids.items()}
        for r in bydb[db['database_index']]:
            launches=[x for x in r['CANN_APIs']if x['name_text']=='launch'];assert len(launches)==1
            launch=launches[0];tid=launch['globalTid'];all_events=tids.get(tid,[]);times=starts.get(tid,[])
            part=all_events[bisect_left(times,launch['startNs']):bisect_right(times,launch['endNs'])]
            children=[x for x in part if x['endNs']<=launch['endNs']]
            counts[len(children)]+=1
            rows.append(dict(database_index=r['database_index'],database_sha256=db['sha256'],
                task_record_rowid=r['TASK']['task_record_rowid'],TASK=r['TASK'],
                exact_CPU_parent=r['exact_connection_CPU_APIs'],runtime_launch=launch,
                same_thread_nested_host_argument_APIs=children,
                relation='Exact TASK runtimeCID identifies one generic launch scope;listed host-argument API is nested within that scope on the same globalTid. This is bounded API nesting,not nearest-event field identity.',
                host_packet_bytes_saved_in_profile=False,native_destination_or_length_recovered=False))
        con.close();after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        dbrefs.append(dict(**identity,database_index=db['database_index'],stat_unchanged=True,mode='ro&immutable=1'))
    assert len(rows)==1272
    write(O/'host-launch-original-API-coverage-v126.json',dict(records=rows,databases=dbrefs,
        nested_host_args_event_count_distribution=dict(counts),original_eager_MemSet_executions=1272,
        source_argument_packet_contents_not_inferred_from_API_name=True,new_device_work=False,overall_complete=False))
    h=N/'sdk-operand-source-v107/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h';lines=h.read_text().splitlines()
    segments=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in [(762,770),(2112,2133),(4345,4355),(5241,5272),(5833,5849)]]
    write(O/'host-launch-installed-ABI-contract-v126.json',dict(header=pin(h),segments=segments,
        host_args='aclrtLaunchKernelWithHostArgs receives hostArgs and byte argsSize;copy only the directly supplied host packet while this call owns it. Do not read any device pointer value embedded inside the packet.',
        placeholders='aclrtPlaceHolderInfo is {uint32 addrOffset,uint32 dataOffset}. The runtime relocates the pointer slot at addrOffset to the copied data starting at dataOffset. Use validated supplied offsets,not a universal fixed3-pointer prefix.',
        other_launch_APIs='Plain aclrtLaunchKernel/V2 use device-resident argument memory;this host-copy rule is specific to the HostArgs entry,not every launch-like pointer.',
        identity='Function handle/name/provider generation,host/run/process/rank/device,thread/stream/task plus capture/update epoch and actual API result must remain. ThreadLastTaskId is limited to the current thread and user-created stream;do not use it as a global correlation ID.',
        packet_bytes_not_tensor_payload_or_write_extent=True,
        runtime_interposition_export_or_live_capture_not_tested=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',original_eager_MemSet_tasks=len(rows),nested_host_args_event_count_distribution=dict(counts),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
