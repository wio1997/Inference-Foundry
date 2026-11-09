#!/usr/bin/env python3
"""Run the unchanged frozen CPU host selector; never imports CANN/Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--include-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.routing-host-',dir=a.output.parent) as td:
        exe=Path(td)/'selector'
        cmd=['c++','-std=c++17','-O2','-I',str(a.include_dir),
             'tools/probe_routing_host_tiling.cpp','-o',str(exe)]
        p=subprocess.run(cmd,text=True,capture_output=True)
        if p.returncode:raise RuntimeError(p.stderr)
        r=subprocess.run([str(exe.resolve())],text=True,capture_output=True,check=True)
        data=json.loads(r.stdout)
    ranges=[]
    for row in data['rows']:
        if not ranges or ranges[-1]['key']!=row[1]:ranges.append(dict(begin=row[0],end=row[0],key=row[1]))
        else:ranges[-1]['end']=row[0]
    selected=[]
    for m in [24,416,816,817,818]:
        row=data['rows'][m-1]
        _,key,cores,per,last,loop,lastloop,nloop,lastnloop,cols,ncols,lastcoreloop=row
        computations=0
        core_computations=[]
        for core in range(cores):
            nrows=last if core==cores-1 else per
            stride=lastcoreloop if core==cores-1 else loop
            chunks=[(core*per,nrows)] if key==21000 else [
                (core*per+i*stride,min(stride,nrows-i*stride)) for i in range((nrows+stride-1)//stride)]
            subtotal=sum((begin+n-1)//6-begin//6+1 for begin,n in chunks if n>0)
            core_computations.append(subtotal)
            computations+=subtotal
        selected.append(dict(M=m,key=key,need_cores=cores,per_core_routes=per,last_core_routes=last,
                             per_loop_routes=loop,per_loop_cols=cols,col_loops=ncols,
                             source_implied_quant_row_computations=computations,unique_input_rows=m,
                             maximum_quant_row_computations_per_core=max(core_computations),
                             boundary_duplicate_rows=computations-m))
    pins=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [
        a.include_dir/'moe_init_routing_quant_v2_tiling.h',a.include_dir/'moe_init_routing_v2_tiling.h',
        a.include_dir/'tiling_base.h',Path('tools/probe_routing_host_tiling.cpp')]]
    result=dict(scope=data['scope'],status='OFFLINE_HOST_SELECTOR_NOT_LOADED_BINARY_ATTESTATION',
        ranges=ranges,selected=selected,source_pins=pins,
        row_fields=['M','tiling_key','need_core_num','per_core_rows','last_core_rows',
                    'per_core_per_loop_rows','per_core_last_loop_rows','per_core_loops',
                    'last_core_loops','per_loop_cols','col_loops','last_core_per_loop_rows'],
        all_rows=data['rows'],
        limits=['Inputs exactly mirror frozen DFC host constants for target24 experts/rank; this is not DSpark topology or a measured native tiling buffer.',
                'Quant-compute counts follow per-core/per-loop 1H source row spans; masks/dropped routes may have no downstream consumer. All original flags and required writes remain separate.'],
        new_NPU_measurements=0)
    a.output.write_text(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','ranges','selected')}))
