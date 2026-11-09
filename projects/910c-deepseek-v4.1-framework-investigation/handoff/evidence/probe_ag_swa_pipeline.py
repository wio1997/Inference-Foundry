#!/usr/bin/env python3
"""Fixed TP8 primitive diagnostic, not a service benchmark or model accuracy test.

Run only inside the user's writable container mount. No tuning sweep, format
conversion, deployment change, global cache fix, or model request is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import traceback


MODEL = Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
KEY = 'layers.0.attn.wkv.weight'
EXPECTED_WEIGHT_SHA = 'dd5b0a5e4e245bd9c3c11931877ba3e342b63f21d4c4e27906ef4b13599c425d'
CASES = [(416, 3328), (818, 6540)]


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def load_weight(torch):
    index = json.loads((MODEL/'quant_model_weights.safetensors.index.json').read_text())
    path = MODEL/index['weight_map'][KEY]
    with path.open('rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(n))
        entry = header[KEY]
        a, b = entry['data_offsets']
        f.seek(8+n+a)
        raw = f.read(b-a)
    assert entry['dtype'] == 'BF16' and entry['shape'] == [512,5120]
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_WEIGHT_SHA
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(512,5120).clone()


def worker(rank, port, work):
    import datetime
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_npu

    work = Path(work)
    out = dict(rank=rank, status='STARTED', checks=[], torch=torch.__version__,
               torch_npu=torch_npu.__version__, weight_sha256=EXPECTED_WEIGHT_SHA)
    path = work/f'rank-{rank}.json'
    save(path, out)
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(rank)
        torch.npu.config.allow_internal_format = False
        out['hf32'] = bool(torch.npu.matmul.allow_hf32)
        # Installed _npuConfig has a write-only setter for this option.
        out['allow_internal_format_request'] = False
        out['device_name'] = torch.npu.get_device_name(rank)
        out['memory_before'] = list(torch.npu.mem_get_info())
        save(path,out)
        assert out['memory_before'][0] >= 2*1024**3, 'Insufficient free memory for isolated diagnostic'
        weight = load_weight(torch).npu()
        dist.init_process_group(backend='hccl', init_method=f'tcp://127.0.0.1:{port}',
                                rank=rank, world_size=8, timeout=datetime.timedelta(seconds=35))
        pg = dist.distributed_c10d._get_default_group()
        hcom = pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        def metrics(x,y):
            a,b=x.cpu().float(),y.cpu().float()
            d=a-b
            rms=float(b.square().mean().sqrt())
            return dict(bit_differences=int((a!=b).sum()), elements=a.numel(),
                        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        max_abs=float(d.abs().max()),reference_RMS=rms,
                        normalized_RMSE=float(d.square().mean().sqrt())/rms if rms else None)

        for local, actual in CASES:
            generator=torch.Generator().manual_seed(20261109+rank)
            cpu=torch.randn((local,5120),generator=generator).to(torch.bfloat16)
            valid=max(0,min(local,actual-rank*local))
            cpu[valid:]=0
            x=cpu.npu()
            gathered=torch.empty((local*8,5120),device=x.device,dtype=x.dtype)
            def baseline():
                dist.all_gather_into_tensor(gathered,x,group=pg)
                return F.linear(gathered[:actual],weight)
            def fused():
                return torch_npu.npu_all_gather_base_mm(
                    x,weight.T,hcom,8,gather_output=True,comm_mode='aiv')
            before=baseline()
            combined,wide=fused()
            torch.npu.synchronize()
            record=dict(local_M=local,actual_M=actual,stock_MM_M=local*8,
                        gather_equal=bool(torch.equal(wide,gathered)),
                        versus_original_trimmed=metrics(combined[:actual],before),
                        versus_padded_linear=metrics(combined,F.linear(gathered,weight)),
                        timing_scope='NPU-event AG+trimmed F.linear versus AIV AGMM (gather_output=True) + view; no concurrent Q or model work')
            # Fixed small diagnostic budget; alternating order, paired barriers
            # outside event timing. New primitive can consume different resources.
            for _ in range(2):
                baseline();fused()
            torch.npu.synchronize()
            samples={'baseline_us':[],'fused_us':[]}
            for repeat in range(7):
                order=[('baseline_us',baseline),('fused_us',fused)]
                if repeat%2:order.reverse()
                for name,fn in order:
                    dist.barrier()
                    start=torch.npu.Event(enable_timing=True)
                    end=torch.npu.Event(enable_timing=True)
                    start.record();value=fn();end.record();end.synchronize()
                    samples[name].append(float(start.elapsed_time(end))*1000)
                    del value
            record.update(samples)
            record['median_delta_us']=statistics.median(samples['baseline_us'])-statistics.median(samples['fused_us'])
            out['checks'].append(record)
            save(path,out)
        out['status']='COMPLETED_PRIMITIVE_DIAGNOSTIC'
        save(path,out)
        dist.destroy_process_group()
    except Exception as e:
        out.update(status='ERROR',error=repr(e),traceback=traceback.format_exc())
        save(path,out)
        raise


if __name__ == '__main__':
    work=Path.cwd().resolve()
    if not str(work).startswith('/data/tiankuan/wio/'):
        raise SystemExit('Diagnostic must run under the user writable mount')
    import torch.multiprocessing as mp
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    manifest=dict(scope='Fixed TP8 real Wkv/synthetic-input primitive diagnostic',
                  cases=CASES,world_size=8,port=port,new_model_requests=0,
                  no_concurrent_Q=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(work/'manifest.json',manifest)
    mp.spawn(worker,args=(port,str(work)),nprocs=8,join=True)
    print(json.dumps({'status':'COMPLETED_PRIMITIVE_DIAGNOSTIC','rank_files':8}))
