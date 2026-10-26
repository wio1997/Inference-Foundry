"""Synthetic EP32 roundtrip; zero model loads or HTTP requests."""
import datetime,hashlib,json,os,pathlib,traceback
import torch
import torch_npu
import torch.distributed as dist
from deep_ep import Buffer
from deep_ep.device_info import get_device_version


def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=180))
    assert dist.get_world_size()==32 and os.environ['DEEP_USE_MODE']=='alltoall'
    site=pathlib.Path(__import__('deep_ep').__file__).parent
    expected=json.loads(os.environ['GLM53_DEEPEP_SOURCE_SHA'])
    for name,sha in expected.items():assert hashlib.sha256((site/name).read_bytes()).hexdigest()==sha,name
    group=dist.group.WORLD
    cfgs=[Buffer.get_dispatch_config(32),Buffer.get_combine_config(32)]
    nvl=max(c.get_nvl_buffer_size_hint(6144*2,32) for c in cfgs)
    rdma=max(c.get_rdma_buffer_size_hint(6144*2,32) for c in cfgs)
    buf=Buffer(group,nvl,rdma,low_latency_mode=False,num_qps_per_rank=Buffer.num_sms)
    assert buf.normal_strategy.get_name()=='alltoall'
    out={'rank':rank,'local_rank':local,'world':32,'device_version':get_device_version(),'strategy':buf.normal_strategy.get_name(),'nvl_bytes':nvl,'rdma_bytes':rdma,'cases':[],'model_requests':0}
    def payload(r,n):
        return (torch.arange(6144,device='npu',dtype=torch.float32).remainder(8)/8+r/32+torch.arange(n,device='npu',dtype=torch.float32)[:,None]/16).to(torch.bfloat16)
    for n in [1,4]:
        for quant in [False,True]:
            label=f'tokens{n}-'+('int8' if quant else 'bf16')
            print(json.dumps({'rank':rank,'begin':label}),flush=True)
            x=payload(rank,n);reference=x.clone()
            idx=(torch.arange(8,device='npu',dtype=torch.int64)+((rank+1)%32)*8).expand(n,8).contiguous()
            weights=torch.full((n,8),1/8,device='npu',dtype=torch.float32)
            a,b,c,d,_=buf.get_dispatch_layout(idx,256)
            recv,ri,rw,counts,handle,event=buf.dispatch(x,num_tokens_per_rank=a,num_tokens_per_rdma_rank=b,num_tokens_per_expert=c,is_token_in_rank=d,topk_idx=idx,topk_weights=weights,use_fp8=quant)
            assert counts==[n]*8,(rank,label,counts)
            if quant:
                q,s=recv;assert q.dtype==torch.int8 and s.dtype==torch.float32
                actual=(q.float()*s[:,None]).to(torch.bfloat16)
            else:
                actual=recv;assert actual.dtype==torch.bfloat16
            expected_recv=payload((rank-1)%32,n).repeat(8,1)
            assert list(actual.shape)==[n*8,6144]
            dispatch_error=(actual.float()-expected_recv.float()).abs().max().item()
            assert dispatch_error<=(0.04 if quant else 0),('dispatch',rank,label,dispatch_error)
            # Combine takes ownership of its input storage; keep an independent reference.
            y,_,_=buf.combine(actual.clone(),handle)
            torch.npu.synchronize()
            combine_error=(y.float()-reference.float()).abs().max().item()
            assert combine_error<=(0.04 if quant else 0),('combine',rank,label,combine_error)
            case={'id':label,'dispatch_max_abs_error':dispatch_error,'combine_max_abs_error':combine_error,'tolerance':0.04 if quant else 0,'recv_shape':list(actual.shape),'expert_counts':counts,'inter_node_exercised':rank in [15,31]}
            out['cases'].append(case)
            dist.barrier()
            print(json.dumps({'rank':rank,'pass':case}),flush=True)
    path=pathlib.Path('/run/glm53')/f'rank-{rank}.json';path.write_text(json.dumps(out,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()
    print('GLM53_DEEPEP_ROUNDTRIP='+json.dumps(out),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
