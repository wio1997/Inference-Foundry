"""Resident native DP2/TP8/EP16 workers; file IPC is under the project root."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True)
    p.add_argument('--control-dir',type=Path,required=True)
    p.add_argument('--tp-size',type=int,default=8)
    p.add_argument('--slots-per-dp',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=34816)
    p.add_argument('--chunk-size',type=int,default=512)
    p.add_argument('--prefix-tokens',type=int,default=29568)
    p.add_argument('--mc2-buffer-mib',type=int,default=2560)
    args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    os.environ['ASCEND_RT_VISIBLE_DEVICES']=str(rank)
    import torch
    import torch.distributed as dist
    import torch_npu
    from deepinfer.npu_ops.dsa910c import load_ops
    from deepinfer.engine.ascend_w8a8_serving import NativeServingRunner
    load_ops()
    torch.npu.set_device(0)
    dist.init_process_group('hccl')
    tp_group=None
    for first in range(0,dist.get_world_size(),args.tp_size):
        group=dist.new_group(list(range(first,first+args.tp_size)),backend='hccl')
        if first<=rank<first+args.tp_size:
            tp_group=group
    if os.environ.get('DEEPINFER_910C_MC2')=='1':
        from deepinfer.npu_ops.mc2_910c import initialize_mc2_group
        initialize_mc2_group(args.mc2_buffer_mib)
    args.control_dir.mkdir(parents=True,exist_ok=True)
    progress=lambda event: print(json.dumps({'rank':rank,**event}),flush=True)
    runner=NativeServingRunner(args.model_dir,tp_group=tp_group,slots_per_dp=args.slots_per_dp,
        prefix_tokens=args.prefix_tokens,device='npu:0',max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,progress=progress)
    dist.barrier()
    if os.environ.get('DEEPINFER_910C_STARTUP_GRAPHS')=='1':
        runner.initialize_dspark_graphs()
    dist.barrier()
    (args.control_dir/f'ready-rank{rank}.json').write_text(json.dumps({'rank':rank,'state':'READY',
        'mc2_capacity_floor':runner.model.layers[0].mc2.capacity_floor,'target_vocab_parallel':runner.model.target_vocab_parallel,
        'target_vocab_start':runner.model.target_vocab_start,
        'target_vocab_width':runner.model.target_vocab_width,
        'dspark_graphs_initialized':bool(runner.model.dspark is not None and
            getattr(runner.model,'_verify_graph',None) is not None and runner.model.dspark.graphs)}))
    previous=None
    while True:
        current=args.control_dir/'current.json'
        if not current.exists():
            time.sleep(.02);continue
        command=json.loads(current.read_text())
        if command['id']==previous:
            time.sleep(.01);continue
        previous=command['id']
        directory=args.control_dir/'jobs'/previous
        job=json.loads((directory/'request.json').read_text())
        try:
            # Frozen experiment controls are copied into the immutable IPC job
            # by the CPU frontend, so every EP rank receives the same mode.
            mode=job.get('single_point_ablation')
            if os.environ.get('DEEPINFER_SINGLE_POINT_ABLATION')=='1':
                if (not isinstance(mode,dict) or set(mode)!={'varlen','tail_ffn','engram_prefetch','engram_pinned','engram_uva'}
                        or any(type(value) is not bool for value in mode.values())):
                    raise ValueError('explicit boolean single-point mode required')
                os.environ['DEEPINFER_ENABLE_MTP_VARLEN']='1' if mode['varlen'] else '0'
                runner.model.tail_ffn=mode['tail_ffn']
                os.environ['DEEPINFER_910C_ENGRAM_PREFETCH']='1' if mode['engram_prefetch'] else '0'
                os.environ['DEEPINFER_910C_ENGRAM_PINNED_STAGING']='1' if mode['engram_pinned'] else '0'
                os.environ['DEEPINFER_910C_ENGRAM_UVA']='1' if mode['engram_uva'] else '0'
                (directory/f'ablation-rank{rank}.json').write_text(json.dumps({
                    'rank':rank,'mode':mode,'all_graphs_retained':True}))
            elif mode is not None:
                raise ValueError('single-point mode requires explicit worker opt-in')
            # Opt-in diagnostic job only. Formal AISBench requests never set
            # profile_npu, so their execution path and metrics stay unchanged.
            if job.get('profile_npu', False) and rank == 0:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True, profile_memory=False, with_stack=False,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(directory/'profile-rank0')),
                ):
                    runner.run(job,directory)
            else:
                runner.run(job,directory)
        except Exception:
            import traceback
            (directory/f'error-rank{rank}.txt').write_text(traceback.format_exc())
            raise


if __name__=='__main__':
    main()
