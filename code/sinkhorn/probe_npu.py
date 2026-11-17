"""Exclusive root-owned Ascend910B component gate, never run on client VM."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,statistics,time,traceback

BASE=Path(__file__).resolve().parent
OUT=BASE/'evidence'


def save(name,d):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)


def original(x,eps=1e-6,iters=20):
    v=x.softmax(-1)+eps
    v=v/(v.sum(-2,keepdim=True)+eps)
    for _ in range(iters-1):
        v=v/(v.sum(-1,keepdim=True)+eps)
        v=v/(v.sum(-2,keepdim=True)+eps)
    return v


def main():
    import torch,torch_npu,triton
    from fused_sinkhorn import fused_sinkhorn
    assert os.environ.get('GLM53_SINKHORN_PROBE_TASK')=='g31', 'Invoke only via exclusive target controller'
    torch.set_num_threads(2);torch.npu.set_device('npu:0');OUT.mkdir(exist_ok=False)
    r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=torch_npu.__version__,triton=triton.__version__,device=torch.npu.get_device_name(0),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'fused_sinkhorn.py',BASE/'mhc_candidate.py',Path(__file__)]},cases=[],graph=[],timing=[],passed=False,scope='Component numerical/graph gate and diagnostic microtiming only; no full-model or AISBench improvement claim.')
    try:
        for t in [0,1,4,16,17,64,128,1024,8192]:
            for family,scale in [('random',1.),('peaked',16.),('zero',0.)]:
                gen=torch.Generator().manual_seed(31000+t)
                x=torch.randn(t,4,4,generator=gen)*scale
                if t:x[0]+=torch.tensor([[0.,1.,2.,3.],[4.,3.,2.,1.],[-3.,1.,5.,-2.],[2.,8.,-4.,0.]])*scale
                device=x.npu();got=fused_sinkhorn(device,1e-6,20);ref=original(device);gold=original(x.double())
                torch.npu.synchronize();cpu=got.cpu()
                torch.testing.assert_close(cpu,ref.cpu(),atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(cpu.double(),gold,atol=2e-6,rtol=2e-5)
                r['cases'].append(dict(tokens=t,family=family,max_abs_vs_native=float((cpu-ref.cpu()).abs().max()) if t else 0.,max_abs_vs_FP64=float((cpu.double()-gold).abs().max()) if t else 0.,passed=True));save('results.json',r)
        for t in [1,16,64,1024]:
            x=torch.randn(t,4,4,device='npu',dtype=torch.float32)
            for _ in range(3):fused_sinkhorn(x,1e-6,20);original(x)
            torch.npu.synchronize();graphs={};outputs={}
            for label,fn in [('baseline',original),('candidate',fused_sinkhorn)]:
                g=torch.npu.NPUGraph()
                with torch.npu.graph(g,auto_dispatch_capture=True):y=fn(x,1e-6,20)
                graphs[label]=g;outputs[label]=y
            x.copy_(torch.randn_like(x)*3);before=torch.npu.memory_allocated();ptr=outputs['candidate'].data_ptr()
            for _ in range(3):
                graphs['baseline'].replay();graphs['candidate'].replay()
            torch.npu.synchronize();torch.testing.assert_close(outputs['candidate'],outputs['baseline'],atol=2e-6,rtol=2e-5)
            assert before==torch.npu.memory_allocated() and ptr==outputs['candidate'].data_ptr()
            r['graph'].append(dict(tokens=t,replays=3,changed_input=True,stable_output_pointer=True,no_replay_allocation=True,passed=True))
            for label in ['baseline','candidate']:
                samples=[]
                for _ in range(3):
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):graphs[label].replay()
                    end.record();torch.npu.synchronize();samples.append(start.elapsed_time(end)*1000/20)
                r['timing'].append(dict(tokens=t,label=label,graph_replay_us=samples,median_us=statistics.median(samples)))
            save('results.json',r)
        # Full adapter arithmetic retains original projections and post mixing.
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
        base=load('baseline_mhc',BASE/'mhc_baseline.py');candidate=load('candidate_mhc',BASE/'mhc_candidate.py');r['adapter_cases']=[]
        for t in [1,4,17,64,1024]:
            gen=torch.Generator().manual_seed(31100+t)
            args=[torch.randn(t,16384,generator=gen).bfloat16().npu(),(torch.randn(24,16384,generator=gen)*.01).npu(),torch.tensor([.8,1.1,.7],device='npu'),(torch.randn(24,generator=gen)*.1).npu()]
            before=base.hc_pre(*args,4,1e-5,1e-6,20);after=candidate.hc_pre(*args,4,1e-5,1e-6,20)
            torch.testing.assert_close(before[0],after[0],atol=0,rtol=0);torch.testing.assert_close(before[2],after[2],atol=0,rtol=0);torch.testing.assert_close(before[1],after[1],atol=2e-6,rtol=2e-5);assert before[3] is after[3] is False
            branch=torch.randn(t,4096,generator=gen).bfloat16().npu();a=base.hc_post(branch,args[0],before[2],before[1],4);b=candidate.hc_post(branch,args[0],after[2],after[1],4)
            torch.testing.assert_close(a,b,atol=.016,rtol=.01)
            r['adapter_cases'].append(dict(tokens=t,layer_input_and_post_coefficients_bit_exact=True,comb_max_abs=float((before[1]-after[1]).abs().max()),post_output_max_abs=float((a.float()-b.float()).abs().max()),passed=True));save('results.json',r)
        r['passed']=True;save('results.json',r)
    except BaseException as exc:
        r['error']=repr(exc);r['traceback']=traceback.format_exc();save('results.json',r);raise


if __name__=='__main__':main()
