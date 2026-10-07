"""Run249 attention/A2A host-supply decomposition; source scopes are not CPU time."""
from pathlib import Path
import bisect,collections,gzip,hashlib,json
from reduce_gap_owners import segments
from reduce_prepare_criticality import duration,intersection


def reduce(data,queue):
    cpu=sorted((int(a),int(b),n) for a,b,n,*_ in data['cpu']);starts=[z[0] for z in cpu]
    mla=[z for z in cpu if z[2]=='vllm::mla_forward'];assert len(mla)==395
    phases=collections.defaultdict(list);a2a=collections.defaultdict(list);kernels=[];counts=collections.Counter();examples=[]
    for index,(a,b,_) in enumerate(mla):
        xs=[z for z in cpu[bisect.bisect_left(starts,a):bisect.bisect_right(starts,b)] if z[1]<=b]
        def one(name):
            out=[z for z in xs if z[2]==name];assert len(out)==1,(index,name,len(out));return out[0]
        sfa=one('_C_ascend::npu_sparse_flash_attention');gather=one('c10d::_allgather_base_');uv=one('_C_ascend::batch_matmul_transpose');tp=one('c10d::allreduce_')
        assert a<gather[0]<gather[1]<=sfa[0]<sfa[1]<=uv[0]<uv[1]<=tp[0]<tp[1]<=b
        phases['QKV_norm_indexer_K_Q_and_gather_prepare'].append((a,gather[0]))
        phases['DCP_gather_submission'].append(gather[:2])
        phases['gather_to_SFA_indexer_remap_wait_restore'].append((gather[1],sfa[0]))
        phases['SFA_native_host_submission'].append(sfa[:2])
        fused=[z for z in xs if z[2]=='vllm::sfa_dcp_a2a_fused'];counts['MLA']+=1
        if fused:
            assert len(fused)==1;v,w,_=fused[0];assert sfa[1]<=v<w<=uv[0]
            counts['decode_Q_gather_A2A']+=1
            phases['SFA_to_A2A_LSE_and_views'].append((sfa[1],v))
            phases['A2A_pack_exchange_combine'].append((v,w))
            phases['A2A_to_V_up_projection_entry'].append((w,uv[0]))
            pack=one('_pack_sfa_dcp_output_lse_kernel');exchange=one('c10d::alltoall_base_');combine=one('_fused_sfa_dcp_lse_combine_kernel')
            assert v<pack[0]<pack[1]<exchange[0]<exchange[1]<combine[0]<combine[1]<=w
            for s,e,name in [(v,pack[0],'pack_validation_allocation_JIT_launcher_to_enqueue'),(*pack[:2],'pack_kernel_enqueue_scope'),(pack[1],exchange[0],'recv_allocation_and_collective_entry'),(*exchange[:2],'alltoall_submission_scope'),(exchange[1],combine[0],'combine_allocation_JIT_launcher_to_enqueue'),(*combine[:2],'combine_kernel_enqueue_scope'),(combine[1],w,'A2A_return_cleanup')]:a2a[name].append((s,e))
            kernels.extend([pack[:2],combine[:2]])
            if len(examples)<2:examples.append(dict(MLA_ordinal=index,MLA_ns=[a,b],SFA=sfa,DCP_gather=gather,A2A=fused[0],pack=pack,exchange=exchange,combine=combine,V_up=uv,TP_sum=tp))
        else:
            counts['mixed_compact_KV_gather_no_A2A']+=1
            phases['SFA_to_V_up_projection_entry'].append((sfa[1],uv[0]))
        phases['V_up_native_host_submission'].append(uv[:2])
        phases['V_up_to_TP_sum_O_proj_and_submit'].append((uv[1],tp[0]))
        phases['TP_output_sum_submission'].append(tp[:2])
        phases['TP_sum_return_to_MLA_exit'].append((tp[1],b))
    assert counts==dict(MLA=395,decode_Q_gather_A2A=316,mixed_compact_KV_gather_no_A2A=79)
    _,_,gaps=segments(data,queue)
    def summary(regions):return {n:dict(count=len(xs),scope_wall_ms=duration(xs),local_pre_enqueue_exposure_ms=intersection(xs,gaps)) for n,xs in regions.items()}
    outer=summary(phases);inner=summary(a2a)
    mla_wall=duration([x[:2] for x in mla]);mla_exposure=intersection([x[:2] for x in mla],gaps)
    assert abs(sum(z['scope_wall_ms'] for z in outer.values())-mla_wall)<1e-6
    assert abs(sum(z['local_pre_enqueue_exposure_ms'] for z in outer.values())-mla_exposure)<1e-6
    for metric in ['scope_wall_ms','local_pre_enqueue_exposure_ms']:assert abs(sum(z[metric] for z in inner.values())-outer['A2A_pack_exchange_combine'][metric])<1e-6
    return dict(rank=data['rank'],db_sha256=data['db_sha256'],counts=dict(counts),MLA_wall_ms=mla_wall,MLA_local_exposure_ms=mla_exposure,stages=outer,nested_A2A=inner,examples=examples,limits=['Existing Run249 profiler ON source intervals/local launch exposure, not CPU runtime or patched/OFF gain.','Nested A2A stages are a subset of outer A2A, do not add to MLA stages.','Python JIT launcher preparation is mixed with validation/allocation before exported kernel enqueue; kernel scope itself is enqueue, not full frontend. Exact exclusive costs are unknown.','DCP output/LSE all-to-all plus rank-weighted combine is mathematically necessary for the current sharded KV algorithm; do not remove as duplicate TP output sum.','H6 does not touch attention/HCCL source. H4 can alter downstream timing; saved trace does not prove patched rank-critical budgets. No new experiment or kernel change.'])


if __name__=='__main__':
    here=Path(__file__).resolve().parent;point=here.parents[1];rows=[]
    for rank in [13,15]:
        paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz']
        row=reduce(*(json.loads(gzip.decompress(p.read_bytes())) for p in paths));row['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};rows.append(row)
        print(rank,row['MLA_local_exposure_ms'],row['stages'],row['nested_A2A'])
    (here/'attention_supply.json').write_text(json.dumps(rows,indent=2)+'\n')
