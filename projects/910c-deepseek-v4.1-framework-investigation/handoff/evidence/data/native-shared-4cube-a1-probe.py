#!/usr/bin/env python3
"""One fixed M416 primitive admission test, not DFC concurrency or model timing."""
import hashlib
import json
from pathlib import Path
import statistics
import struct
import traceback


def save(data):
    Path('result.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    import torch
    import torch_npu
    assert str(Path.cwd()).startswith('/data/tiankuan/wio/')
    torch.set_num_threads(1)
    result=dict(scope='layer0 real shared weights, synthetic BF16 M416, single device0, fixed4AIC8AIV; no DFC, no service requests',checks={},new_service_requests=0)
    save(result)
    try:
        torch.npu.set_device(0)
        result.update(torch=torch.__version__,torch_npu=torch_npu.__version__,memory_before=list(torch.npu.mem_get_info()),hf32=bool(torch.npu.matmul.allow_hf32))
        assert result['memory_before'][0]>2*1024**3,'Memory guard'
        model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
        idx=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
        expected=json.loads(Path('weight-contract.json').read_text())
        def load(name):
            k='layers.0.ffn.shared_experts.'+name
            with (model/idx[k]).open('rb') as f:
                n,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(n));e=h[k];a,b=e['data_offsets'];f.seek(8+n+a);raw=f.read(b-a)
            sha=hashlib.sha256(raw).hexdigest();assert sha==expected[k]['sha256']
            dtype={'I8':torch.int8,'F32':torch.float32}[e['dtype']]
            return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(e['shape']).clone()
        wup=torch.cat([load('w1.weight'),load('w3.weight')],dim=0).T.contiguous().npu()
        wdown=load('w2.weight').T.contiguous().npu()
        sup=torch.cat([load('w1.weight_scale'),load('w3.weight_scale')],dim=0).flatten().npu()
        sdown=load('w2.weight_scale').flatten().npu()
        result['offsets_all_zero']={n:bool((load(n+'.weight_offset')==0).all()) for n in ['w1','w2','w3']}
        torch.ops.load_library('/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so')
        gen=torch.Generator().manual_seed(20261109)
        x=torch.randn((416,5120),generator=gen).to(torch.bfloat16).npu()
        def dq():return torch_npu.npu_dynamic_quant(x)
        def up(q,w):return torch_npu.npu_quant_matmul(q,w,sup,pertoken_scale=None,bias=None,output_dtype=torch.int32)
        def act(z,s):return torch.ops._C_ascend.npu_dequant_swiglu_quant(x=z,weight_scale=sup,activation_scale=s,bias=None,quant_scale=None,quant_offset=None,group_index=None,activate_left=True,quant_mode=1,swiglu_mode=1,clamp_limit=10.,glu_alpha=1.,glu_bias=0.)
        def down(q,s,w):return torch_npu.npu_quant_matmul(q,w,sdown,pertoken_scale=s,bias=None,output_dtype=torch.bfloat16)
        def bits(a,b):
            aa,bb=a.cpu().contiguous(),b.cpu().contiguous()
            # Compare bytes, including signed zero and NaN payloads.
            return dict(byte_differences=int((aa.view(torch.uint8)!=bb.view(torch.uint8)).sum()),elements=aa.numel(),dtype=str(aa.dtype),finite=bool(torch.isfinite(aa.float()).all() and torch.isfinite(bb.float()).all()))
        q,s=dq();u=up(q,wup);a,sa=act(u,s);d=down(a,sa,wdown)
        torch.npu.synchronize()
        result['ND_default_complete']=True;save(result)
        nz=None
        try:
            # Baseline only: no stream limits are applied to these NZ operators.
            torch.npu.config.allow_internal_format=True
            wu_nz=torch_npu.npu_format_cast(wup,29);wd_nz=torch_npu.npu_format_cast(wdown,29)
            un=up(q,wu_nz);dn=down(a,sa,wd_nz);torch.npu.synchronize()
            nz=(un,dn)
            result['NZ_default_complete']=True
            result['NZ_vs_ND']={'Up':bits(un,u),'Down_same_activation':bits(dn,d)}
        except Exception as exc:
            result.update(NZ_default_complete=False,NZ_error=repr(exc),NZ_traceback=traceback.format_exc())
        finally:
            # Installed setting is write-only; record request, never a getter.
            torch.npu.config.allow_internal_format=False
        save(result)
        limited=torch.npu.Stream()
        torch.npu.set_stream_limit(limited,cube_num=4,vector_num=8)
        result['requested_limit']={'cube':4,'vector':8,'allow_internal_format':False}
        result['get_stream_limit']=torch.npu.get_stream_limit(limited)
        limited.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(limited):
            q4,s4=dq();u4=up(q,wup);a4,sa4=act(u,s);d4=down(a,sa,wdown)
        limited.synchronize()
        result['checks']={'DQ':bits(q,q4),'DQ_scale':bits(s,s4),'Up_INT32':bits(u,u4),'Activation_INT8':bits(a,a4),'Activation_scale':bits(sa,sa4),'Down_BF16_same_input':bits(d,d4)}
        if nz is not None:
            result['checks'].update(NZ_Up_to_limited_ND=bits(nz[0],u4),NZ_Down_to_limited_ND_same_input=bits(nz[1],d4))
        result['all_tested_bit_equal']=all(c['byte_differences']==0 and c['finite'] for c in result['checks'].values())
        save(result)
        # A bit counterexample is sufficient to stop this direct stock recipe.
        if result['all_tested_bit_equal']:
            def pipeline():
                qq,ss=dq();uu=up(qq,wup);aa,saa=act(uu,ss);return down(aa,saa,wdown)
            for stream in [torch.npu.current_stream(),limited]:
                with torch.npu.stream(stream):pipeline();pipeline()
            torch.npu.synchronize()
            samples={'ND_default_us':[],'ND_4_8_us':[]}
            for rep in range(7):
                order=[('ND_default_us',torch.npu.current_stream()),('ND_4_8_us',limited)]
                if rep%2:order.reverse()
                for key,stream in order:
                    with torch.npu.stream(stream):
                        st=torch.npu.Event(enable_timing=True);en=torch.npu.Event(enable_timing=True)
                        st.record();v=pipeline();en.record();en.synchronize()
                        samples[key].append(float(st.elapsed_time(en))*1000)
            result['isolated_pipeline_timing']=dict(samples=samples,medians={k:statistics.median(v) for k,v in samples.items()},scope='ND only, no DFC or requests; admission diagnostic, no speedup claim')
            save(result)
            try:
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(Path.cwd()/'prof')),record_shapes=True,with_stack=False) as prof:
                    with torch.npu.stream(limited):pipeline()
                    limited.synchronize()
                    prof.step()
                result['profile_complete']=True
            except Exception as exc:result.update(profile_complete=False,profile_error=repr(exc),profile_traceback=traceback.format_exc())
        torch.npu.reset_stream_limit(limited)
        result['status']='BIT_COUNTEREXAMPLE' if not result['all_tested_bit_equal'] else 'TESTED_PRIMITIVES_BIT_EQUAL_ONLY'
    except Exception as exc:
        result.update(status='ERROR',error=repr(exc),traceback=traceback.format_exc())
    finally:
        save(result)
    print(json.dumps(dict(status=result['status'],checks=result['checks'])))


if __name__=='__main__':main()
