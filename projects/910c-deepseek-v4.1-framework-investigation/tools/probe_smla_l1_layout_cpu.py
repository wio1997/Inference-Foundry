#!/usr/bin/env python3
"""Bound source-address/NZ layout claims; no accelerator or arithmetic emulation."""
from pathlib import Path
from hashlib import sha256
import json,struct,subprocess,importlib.util,zlib

OUT=Path('audit/execution-search/smla-l1-layout-cpu-a1')
ROOT=Path('audit/execution-search')
OLD=Path('/Users/wio/work/branch-fusion-audit-20261104')

def digest(p):return sha256(p.read_bytes()).hexdigest()
def th(rows):return sha256(b''.join(struct.pack('<'+'i'*len(r),*r) for r in rows)).hexdigest()
def nz(r,d,n):return d//16*n*16+r*16+d%16
def align(n):return (n+15)//16*16

def main():
 OUT.mkdir(exist_ok=False)
 spec=importlib.util.spec_from_file_location('probe','tools/probe_smla_transfer_plan_cpu.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
 header=ROOT/'smla-l1-runtime-source/usr/local/Ascend/cann-9.2.0/aarch64-linux/asc/include/interface/kernel_struct_data_copy.h'
 text=header.read_text();a=text.index('struct Nd2NzParams {');b=text.index('\n};',a)+3
 code='#include <cstdint>\n#include <iostream>\n#define __aicore__\n'+text[a:b]+r'''
int main(){Nd2NzParams p;uint64_t x=127*512,y=128*512;p.srcDValue=x;std::cout<<sizeof(p.srcDValue)<<" "<<p.srcDValue<<" ";p.srcDValue=y;std::cout<<p.srcDValue<<"\n";}
'''
 (OUT/'field.cpp').write_text(code);field=[]
 for arch in [2201,3510]:
  binary=OUT/f'field-{arch}';build=subprocess.run(['clang++','-std=c++17',f'-D__NPU_ARCH__={arch}',str(OUT/'field.cpp'),'-o',str(binary)],capture_output=True,text=True);(OUT/f'build-{arch}.log').write_text(build.stdout+build.stderr);assert build.returncode==0
  value=subprocess.check_output([str(binary)],text=True).strip();assert value==('2 65024 0' if arch==2201 else '8 65024 65536');field.append(dict(arch=arch,stdout=value))
 qli=json.loads((OLD/'full-review/QLI-CANDIDATE-NATIVE.json').read_text());seen=set();fixtures=[];pins=[]
 for rel in qli['native_receipts']:
  p=OLD/'full-review'/rel;assert digest(p)==qli['input_pins'][rel];pins.append(dict(path=str(p),sha256=digest(p)));receipt=json.loads(p.read_text())
  for name in ['empty_packet','mixed_packet','full_packet_boundary']:
   item=receipt['result']['checks'][name]['baseline'];assert th(item['raw_ids'])==item['raw_sha256']
   prepared=[sorted(i for i in row if 0<=i<lim) for row,lim in zip(item['raw_ids'],item['visible'])]
   assert list(map(len,prepared))==item['lengths'];padded=[r+[-1]*(512-len(r)) for r in prepared];assert th(padded)==item['prepared_sha256']
   if item['prepared_sha256'] in seen:continue
   seen.add(item['prepared_sha256'])
   for row,(ids,lim) in enumerate(zip(padded,item['visible'])):
    np=(lim+127)//128
    for reverse in [False,True]:fixtures.append(dict(name=f"{item['prepared_sha256']}/r{row}/reverse{reverse}",ids=ids,limit=lim,pages=list(range(np,0,-1)) if reverse else list(range(1,np+1)),block=128,K=sum(i>=0 for i in ids),seed=0,ring=0))
 parts=[str(len(fixtures))]
 for c in fixtures:parts.extend([f"128 {c['limit']} 0 0 {len(c['pages'])}",' '.join(map(str,c['pages'])),' '.join(map(str,c['ids']))])
 driver=ROOT/'smla-transfer-plan-cpu-a2/driver';run=subprocess.run([str(driver)],input='\n'.join(parts)+'\n',capture_output=True,text=True,timeout=30);assert run.returncode==0,run.stderr;traces=json.loads(run.stdout)
 rows=[];active_coordinates=0
 for c,trace in zip(fixtures,traces):
  actual=dict(trace);expected,_=probe.emit(c['ids'],c['pages'],128,c['limit']);assert actual==expected
  assert all(i in actual for i in range(c['K']));groups=probe.cover([actual[i] for i in range(c['K'])]);last=(c['K']-1)//128*128 if c['K'] else 0
  rows.append(dict(name=c['name'],K=c['K'],visible=c['limit'],groups=len(groups),direct_instructions=6*len(groups),old_dense_instructions=6*((c['K']+127)//128),retained_tail_rows=c['K']-last,retained_instructions=6*len(groups)-4*sum(g['dest']>=last for g in groups)))
 # Check active coordinate source provenance, including differently shaped QK and PV L1 tiles.
 for K in [0,1,2,3,15,16,17,63,64,65,127,128,129,255,256,257,511,512]:
  for tile in range(0,K,128):
   nr=min(128,K-tile);na=align(nr);qk=[]
   for half in range(2):
    layout={nz(r,d,na):(tile+r)*512+half*256+d for r in range(nr) for d in range(256)};assert len(layout)==nr*256;qk.append(layout)
   for strip in range(4):
    # PV has two128x128 subtiles per256-K work slot, not one QK128x256 panel.
    pv_base=(tile//128%2)*128*128
    direct={pv_base+nz(r,d,na):(tile+r)*512+strip*128+d for r in range(nr) for d in range(128)}
    for r in range(nr):
     for d in range(128):
      retained=qk[strip//2][nz(r,strip%2*128+d,na)]
      assert direct[pv_base+nz(r,d,na)]==retained;active_coordinates+=1
 # Functional role-lifetime witnesses only. Async flags/prefetch throughput unproved.
 schedules=[]
 for ori in [1,2,4,8]:
  tasks=[(q,kind) for q in range(4) for kind in ['ori']*ori+['cmp']];pin=None;events=[]
  for i in range(len(tasks)+1):
   cur=tasks[i] if i<len(tasks) else None;prev=tasks[i-1] if i else None
   if cur:
    if cur[1]=='cmp':assert pin is None;pin=cur[0];events.append(['QK_pin_two',pin])
    else:events.append(['QK_work_slot',cur[0]])
   if prev:
    if prev[1]=='cmp':assert pin==prev[0];events.append(['PV_read_pin_release',pin]);pin=None
    else:events.append(['PV_work_slot',prev[0]])
  assert pin is None;schedules.append(dict(ori_tasks_per_query=ori,events=events))
 result=dict(status='PASS_CPU_FIELDS_AND_ACTIVE_NZ_SOURCE_COORDINATES_NOT_NPU_BIT_GATE',field_source_sha256=digest(header),field_compilation=field,original_driver_sha256=digest(driver),native_QLI_receipt_pins=pins,prepared_output_hashes=sorted(seen),native_fixture_PA_scenarios=rows,active_QK_PV_source_coordinates_compared=active_coordinates,role_lifetimes=schedules,limits=['NativeQLI saved prepared IDs are reused; PA ascending/reverse are scenarios, not the model cache map.','NZ mapping is the scalar coordinate interpretation of existing C0/N stride source. Actual ND2NZ N padding, hardware LoadData3D transpose/tail and MMAD outputs are NOT tested.','Retaining the last <=128 rows requires two64KiB QK KV slots and using the third64KiB slot for intermediate QK/PV. Existing4QP slots256KiB unchanged;448KiB total source allocation.','Functional task order is QK(i), PV(i-1); true async flags, MTE completion, reduced prefetch, host keys and all branches remain unproved. Adjacent CMP tasks or unsupported legal branches must fallback; no code implementation exists.','A5 field compilation is only a contrast to the installed shared header, not a deployed A5 path.'],new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],native_scenarios=len(rows),coordinate_checks=active_coordinates,instruction_ranges=[min(r['direct_instructions'] for r in rows),max(r['direct_instructions'] for r in rows)],retained_ranges=[min(r['retained_instructions'] for r in rows),max(r['retained_instructions'] for r in rows)])))

if __name__=='__main__':main()
