"""Bounded real HCCL/default-stream event dependency tests; no model weights."""
import ast
import importlib
import json
import os
import socket
import sys
import types
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


def bits(x):return x.detach().cpu().contiguous().view(torch.uint8)


def worker(rank,port,destination):
    torch.set_num_threads(1);torch.npu.set_device(rank)
    os.environ['GLM_EVENT_WITNESS_SCOPE']='native'
    u=importlib.import_module('vllm_ascend.utils')
    config=types.SimpleNamespace(multistream_overlap_shared_expert=False)
    u.get_ascend_config=lambda:config
    for name in ['fused_moe','moe_comm_method','moe_mlp']:
        module=importlib.import_module('vllm_ascend.ops.fused_moe.'+name)
        assert module.maybe_record_moe_event is u.maybe_record_moe_event
    shared=importlib.import_module('vllm_ascend.ops.fused_moe.shared_experts')
    assert 'before_routed_experts' in shared.FusedMoEEvents.__annotations__
    source=Path(shared.__file__).read_text();tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AscendSharedExperts')
    fwd=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
    wait=next(x for x in fwd.body if isinstance(x,ast.FunctionDef) and x.name=='maybe_wait_event')
    ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[wait],type_ignores=[])),'real_candidate_wait','exec'),ns)
    wait=ns['maybe_wait_event']
    dist.init_process_group('hccl',init_method='tcp://127.0.0.1:%d'%port,rank=rank,world_size=16)
    root=Path(destination).parent;mode=root/'event_mode.bin'
    def switch(value):
        dist.barrier()
        if rank==0:
            fd=os.open(mode,os.O_WRONLY);assert os.pwrite(fd,bytes([value]),0)==1;os.fsync(fd);os.close(fd)
        dist.barrier()
    def flow(local):
        ev0=u.maybe_record_moe_event()
        pieces=[torch.empty_like(local) for _ in range(16)];dist.all_gather(pieces,local)
        ev1=u.maybe_record_moe_event()
        routed=torch.cat(pieces,dim=0)*2
        ev2=u.maybe_record_moe_event();routed=routed+3
        ev3=u.maybe_record_moe_event()
        wait(ev0);part=local+1
        wait(ev2);part=part*2
        wait(ev3);dist.all_reduce(part)
        return routed+part.repeat(16,1),[ev0,ev1,ev2,ev3]
    rows=[];retained=[]
    for n in (1,3,16,17,32):
        for dtype in (torch.float16,torch.bfloat16):
            cpu=torch.full((n,64),rank+1.,dtype=dtype);local=cpu.to('npu');before=bits(local)
            switch(0);a,events_a=flow(local);switch(1);b,events_b=flow(local)
            expected=torch.cat([torch.full((n,64),r+1.,dtype=dtype) for r in range(16)])*2+3
            expected=expected+torch.full_like(expected,304.)
            equal=torch.equal(bits(a),bits(b));exact=torch.equal(bits(b),expected.view(torch.uint8))
            preserved=torch.equal(before,bits(local));lifetime=all(torch.equal(bits(x),s) for x,s in retained[-2:])
            policy=all(e is not None for e in events_a) and all(e is None for e in events_b)
            ok=equal and exact and preserved and lifetime and policy
            flag=torch.tensor([int(ok)],dtype=torch.int32,device='npu');dist.all_reduce(flag,op=dist.ReduceOp.MIN);globally=bool(flag.item())
            rows.append(dict(rows=n,dtype=str(dtype),stock_candidate_bytes_equal=equal,independent_expected_bytes=exact,
                             input_bytes_preserved=preserved,retained_output_lifetime=lifetime,event_policy=policy,all_ranks_passed=globally))
            retained.append((b,bits(b)))
    # Correctness of the retained true branch, using an actual auxiliary stream.
    config.multistream_overlap_shared_expert=True;switch(1)
    x=torch.arange(128,dtype=torch.float32,device='npu');x=x*2;event=u.maybe_record_moe_event();assert event is not None
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):wait(event);y=x+7
    torch.npu.current_stream().wait_stream(stream)
    ok=torch.equal(bits(y),(torch.arange(128,dtype=torch.float32)*2+7).view(torch.uint8))
    flag=torch.tensor([int(ok)],dtype=torch.int32,device='npu');dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    rows.append(dict(overlap_true_cross_stream=True,all_ranks_passed=bool(flag.item())))
    torch.npu.synchronize();Path(destination,'rank%d.json'%rank).write_text(json.dumps(dict(rank=rank,cases=rows,passed=all(x['all_ranks_passed'] for x in rows)),indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    destination,ownerpath=sys.argv[1:];Path(destination).mkdir(parents=True,exist_ok=True)
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    Path(ownerpath).write_text(json.dumps(dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=stat[19],state=stat[0],ppid=int(stat[1])))+'\n')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,destination),nprocs=16,join=True)
    rows=[json.loads(Path(destination,'rank%d.json'%r).read_text()) for r in range(16)];passed=all(x['passed'] for x in rows)
    out=dict(passed=passed,ranks=16,cases_per_rank=11,actual_model_loaded=False,real_installed_candidate_imports=True)
    Path(destination,'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True);sys.exit(0 if passed else 1)
