#!/usr/bin/env python3
"""Replay original DFC host DoTiling on CPU and enumerate quantization row owners."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,subprocess
OUT=Path('audit/execution-search')
def main():
    dep=json.loads((OUT/'dfc-input-host-dependencies.json').read_text());pins=[]
    for r in dep['sources']:
        p=Path(r['local_path']);h=sha256(p.read_bytes()).hexdigest();assert h==r['runtime_sha256'];pins.append(dict(path=str(p),sha256=h))
    header=Path(next(r['local_path'] for r in dep['sources'] if r['runtime_path'].endswith('/moe_init_routing_quant_v2_tiling.h')))
    dst=OUT/'dfc-input-host-cpu';dst.mkdir(exist_ok=True)
    driver=dst/'driver.cpp'
    code='''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "'''+str(header)+'''"
int main(){
 for(int64_t m: {0,3,6,24,416,540,544,816,817,818,3328,6540}){
  optiling::MoeInitRoutingQuantV2TilingBase t;
  bool ok=t.DoTiling(m,5120,6,0,385,0,0,2,false,2,1,0,40,196352);
  auto &g=t.quantTilingData.gatherOutComputeParamsOp;
  std::printf("%lld %d %llu %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\\n",(long long)m,ok,(unsigned long long)t.tilingKey_,(long long)t.quantTilingData.smoothType,(long long)g.needCoreNum,(long long)g.perCoreRows,(long long)g.lastCoreRows,(long long)g.perCorePerLoopRows,(long long)g.lastCorePerLoopRows,(long long)g.perCoreLoops,(long long)g.lastCoreLoops,(long long)g.perCoreLastLoopRows,(long long)g.lastCoreLastLoopRows,(long long)g.colLoops);
 }
}
'''
    driver.write_text(code);exe=dst/'driver'
    cmd=['c++','-std=c++17','-O0',str(driver),'-o',str(exe)]
    build=subprocess.run(cmd,capture_output=True,text=True)
    if build.returncode:
        (OUT/'dfc-input-host-cpu-build-invalid.json').write_text(json.dumps(dict(command=cmd,stdout=build.stdout,stderr=build.stderr,driver_sha256=sha256(driver.read_bytes()).hexdigest()),indent=2)+'\n')
        raise RuntimeError(build.stderr)
    run=subprocess.run([str(exe.resolve())],capture_output=True,text=True,check=True)
    keys=['M','ok','host_source_key','smooth_type','need_cores','per_core_expanded_rows','last_core_expanded_rows','per_loop_rows','last_core_per_loop_rows','core_loops','last_core_loops','last_loop_rows','last_core_last_loop_rows','col_loops']
    cases=[]
    for line in run.stdout.splitlines():
        r=dict(zip(keys,map(int,line.split())));assert len(line.split())==len(keys);assert r['ok']==1
        seen=Counter();segments=[];total=6*r['M']
        for core in range(r['need_cores']):
            final=core==r['need_cores']-1
            start=core*r['per_core_expanded_rows'];length=r['last_core_expanded_rows'] if final else r['per_core_expanded_rows']
            loops=r['last_core_loops'] if final else r['core_loops'];step=r['last_core_per_loop_rows'] if final else r['per_loop_rows'];tail=r['last_core_last_loop_rows'] if final else r['last_loop_rows']
            # FullLoad uses one segment/core; gathered 1H uses its row loops.
            if r['host_source_key']==21000:loops=1;step=length;tail=length
            assert r['col_loops']==1 or r['M']==0
            for loop in range(loops):
                a=start+loop*step;n=tail if loop==loops-1 else step;b=a+n
                assert n>0 and 0<=a<b<=total
                input_rows=list(range(a//6,(b-1)//6+1))
                seen.update(input_rows);segments.append(dict(core=core,loop=loop,expanded_start=a,expanded_end=b,input_first=input_rows[0],input_last=input_rows[-1]))
        assert set(seen)==set(range(r['M']))
        calls=sum(seen.values());r.update(quant_compute_calls=calls,unique_input_rows=r['M'],duplicate_calls=calls-r['M'],compute_calls_per_unique=calls/r['M'] if r['M'] else None,owner_multiplicity_histogram=dict(Counter(seen.values())),segments=segments,source_BF16_input_logical_bytes=calls*5120*2,expanded_INT8_scale_bytes=total*(5120+32),unique_codes_scale_logical_bytes=r['M']*(5120+32))
        cases.append(r)
    result=dict(status='CPU_ORIGINAL_HOST_DOTILING_AND_ROW_OWNERSHIP_ONLY',cases=cases,compiler_command=cmd,compiler_stdout=build.stdout,compiler_stderr=build.stderr,run_stdout=run.stdout,driver_path=str(driver.resolve()),driver_sha256=sha256(driver.read_bytes()).hexdigest(),executable_sha256=sha256(exe.read_bytes()).hexdigest(),source_pins=pins,scope='Original host headers executed locally on CPU using source attrs K5120 topK6 expertNum385 UB196352 AIV40 smoothNone. Not active device key, performance, quantization numeric or transport proof. M0 is host empty diagnostic only, not selected legal device dispatch.',findings=['Within each core/loop quant is shared before TopK copies, but a row crossing cores/loops is recomputed. For actual local M3 the18 expanded rows get18 cores and18 Compute calls: six per original row. For illustrative M24,48 calls for24 rows. Original prefill local416 has436 calls; local816/817/818 have836/837/838 calls.','Source original FullLoad21000 and Gather11000/11010 selection changes with M; never apply universal unique-once or TopK-times factor. Source key not active loaded tiling receipt.','Deduplicating arithmetic alone retains six expanded records unless transport/receiver ownership is also redesigned. Original Div, FP32-to-half TRUNC then int8 ROUND, scale and zero/special behavior must all remain bitwise.'],new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (OUT/'dfc-input-host-cpu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['segments']} for r in cases],ensure_ascii=False))
if __name__=='__main__':main()
