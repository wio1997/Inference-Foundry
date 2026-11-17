"""Real EP8 INT8 LL T64 graph gate: fresh process, optional default-stream prewarm."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import traceback
from native_pg_options import get_torch_distributed_pg_options
from reference import WORLD,EXPERTS,LOCAL_EXPERTS,TOPK,HIDDEN,route,weights,qrow,exponent,expected_counts

def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)

def main():
    import torch,torch_npu,torch.distributed as dist,deep_ep,deep_ep_cpp
    rank=int(os.environ['RANK']);os.sched_setaffinity(0,{168+rank});assert rank==int(os.environ['LOCAL_RANK']) and int(os.environ['WORLD_SIZE'])==8
    out=Path(os.environ['LL_GRAPH_OUT']);out.mkdir(parents=True,exist_ok=True);path=out/f'rank{rank}.json';assert not path.exists()
    prewarm=False;wire=os.environ['LL_GRAPH_WIRE']
    record=dict(rank=rank,pid=os.getpid(),prewarm_HCCL_world=prewarm,wire=wire,HCCL_BUFFSIZE=os.environ['HCCL_BUFFSIZE'],phase='imports',scope='DeepSeekV41E384H5120EP8nativeLLgraphandordinaryHCCLbuffer-splitcomponent;notfullmodel/performance')
    save(path,record)
    try:
        root=Path(deep_ep.__file__).resolve().parent
        checks=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
        for name,digest in checks.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
        from native_pg_options import get_torch_distributed_pg_options
        dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=120),pg_options=get_torch_distributed_pg_options())
        cpu_group=dist.new_group(backend='gloo');cap=128;n=80
        torch.npu.synchronize();torch.npu.empty_cache();record['memory_before_groups']=torch.npu.mem_get_info()[0];groups=[];ep=None;tp=None
        for name,group in [('DEFAULT',dist.group.WORLD)]+[(name,dist.new_group(list(range(8)),backend='hccl',pg_options=get_torch_distributed_pg_options(name))) for name in ['world','tp','moe_ep']]:
            before=torch.npu.mem_get_info()[0];value=torch.ones(1,dtype=torch.float32,device=f'npu:{rank}');dist.all_reduce(value,group=group);copies=[torch.empty_like(value) for _ in range(8)];dist.all_gather(copies,value,group=group);torch.npu.synchronize();assert value.item()==8 and all(t.item()==8 for t in copies);groups.append({'name':name,'free_before':before,'free_after':torch.npu.mem_get_info()[0]})
            if name=='moe_ep':ep=group
            if name=='tp':tp=group
        record.update(groups=groups,DEEPEP_HCCL_BUFFSIZE=os.environ['DEEPEP_HCCL_BUFFSIZE'],master_port=os.environ.get('MASTER_PORT'));save(path,record)
        large=torch.full((8192,5120),rank+1.,dtype=torch.bfloat16,device=f'npu:{rank}');base_large=large.clone();dist.all_reduce(large,group=tp);torch.npu.synchronize();assert bool((large==36).all().item())
        graph_tp=torch.npu.NPUGraph()
        with torch.npu.graph(graph_tp,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):
            reduced=base_large.clone();dist.all_reduce(reduced,group=tp)
        for factor in [1,2,3]:
            base_large.fill_(float((rank+1)*factor));dist.barrier(group=cpu_group);graph_tp.replay();torch.npu.synchronize();assert bool((reduced==36*factor).all().item())
        record['large_TP_8192x5120_BF16_graph_three_changed_replays']=True;del graph_tp,large,base_large,reduced;torch.npu.synchronize();torch.npu.empty_cache();record['memory_after_groups_and_TP']=torch.npu.mem_get_info()[0];save(path,record)
        hint=deep_ep.Buffer.get_low_latency_rdma_size_hint(cap,HIDDEN,WORLD,EXPERTS)
        buf=deep_ep.Buffer(ep,num_nvl_bytes=0,num_rdma_bytes=hint,low_latency_mode=True,num_qps_per_rank=LOCAL_EXPERTS,normal_strategy='default',low_latency_strategy='default')
        qc=torch.tensor([qrow(rank,t) for t in range(n)],dtype=torch.int8)
        scale=torch.tensor([2.0**exponent(rank,t) for t in range(n)],dtype=torch.float32)
        xc=(qc.float()*scale[:,None]).bfloat16();wc=torch.tensor([weights(rank,t) for t in range(n)],dtype=torch.float32)
        x=xc.to(f'npu:{rank}');ids=torch.tensor([route(rank,t) for t in range(n)],dtype=torch.int32,device=x.device);w=wc.to(x.device)
        gold=(xc.float()*wc.sum(-1,keepdim=True)).bfloat16().float()
        record.update(extension_sha256=hashlib.sha256(Path(deep_ep_cpp.__file__).read_bytes()).hexdigest(),hint_bytes=hint,phase='buffers_ready');save(path,record)

        def forward():
            recv,counts,handle,event,hook=buf.low_latency_dispatch(x,ids,cap,EXPERTS,use_fp8=(wire=='int8'),use_mxfp4=False,use_mxfp8=False,topk_weights=w,async_finish=False,return_recv_hook=False)
            if wire=='int8':
                q,s=recv;expert=(q.float()*s.reshape(q.shape[:-1]+(1,))).bfloat16()
            else:expert=recv
            combined,event,hook=buf.low_latency_combine(expert,ids,w,handle,async_finish=False,return_recv_hook=False,use_mxfp8=False)
            return combined,counts

        if prewarm:
            record['phase']='explicit_HCCL_preinit';save(path,record)
            tiny=torch.zeros(1,device=x.device);peers=[torch.empty_like(tiny) for _ in range(8)]
            dist.all_gather(peers,tiny,group=dist.group.WORLD);torch.npu.synchronize()
            record['HCCL_preinit_completed']=True;save(path,record)
        stream=torch.npu.Stream(device=rank);stream.wait_stream(torch.npu.current_stream(rank))
        with torch.npu.stream(stream):
            record['phase']='nondefault_two_warmups';save(path,record)
            for _ in range(2):
                dist.barrier(group=cpu_group);z,c=forward();torch.npu.synchronize()
                assert c.cpu().long().tolist()==expected_counts(rank,n)
                torch.testing.assert_close(z.float().cpu(),gold,rtol=.02,atol=.02)
            record['phase']='capture';save(path,record)
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),stream=stream,auto_dispatch_capture=True):
                got,counts=forward()
            record['phase']='replay';save(path,record)
            errors=[]
            for _ in range(3):
                graph.replay();torch.npu.synchronize();actual=got.float().cpu()
                assert counts.cpu().long().tolist()==expected_counts(rank,n)
                torch.testing.assert_close(actual,gold,rtol=.02,atol=.02);errors.append(float((actual-gold).abs().max()))
        record.update(phase='passed',replays=3,max_abs_errors=errors,memory_after_LL=torch.npu.mem_get_info()[0]);save(path,record);dist.destroy_process_group()
    except BaseException as exc:
        record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save(path,record);raise

if __name__=='__main__':main()
