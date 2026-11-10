#!/usr/bin/env python3
"""Run original scalar metadata planner for CP64 versus global head8, no NPU ABI claim."""
from pathlib import Path
from hashlib import sha256
import ast,json,subprocess

ROOT=Path('audit/execution-search');OUT=ROOT/'attention-head-schedule-cpu-a3'
def main():
 old=Path('tools/probe_smla_schedule_cpu.py').read_text();tree=ast.parse(old)
 prefix=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets))
 src=ROOT/'smla-runtime-source/metadata-repo/sparse_flash_mla_metadata_aicpu.cpp';hdr=src.with_suffix('.h');meta=ROOT/'smla-runtime-source/repo/op_kernel/sparse_flash_mla_kernel_metadata.h'
 text=src.read_text();begin=text.index('int32_t SparseFlashMlaMetadataCpuKernel::GetSumOfQuerySeq()');end=text.index('\nnamespace {\nstatic const char *kernelType');body=text[begin:end]
 h='\n'.join(l for l in hdr.read_text().splitlines() if not any(x in l for x in ['"cpu_context.h"','"cpu_kernel.h"','"cpu_tensor.h"','"../../sparse_flash_mla/','"../../common/'])).replace('private:','public:')
 main_cpp=r'''
using namespace aicpu;
void run(int heads,int perReq,int rank,bool cmp,int ratio,bool consistency) {
 const int B=4,total=B*perReq,perRank=(total+7)/8;
 const int begin=rank<0?0:rank*perRank,finish=rank<0?total:std::min(total,begin+perRank);
 Tensor qs,ori,cm,out;qs.data.push_back(0);int mx=0;
 for(int b=0;b<B;b++) {
  int lo=std::max(begin,b*perReq),hi=std::min(finish,(b+1)*perReq),q=std::max(hi-lo,0);
  qs.data.push_back(qs.data.back()+q);mx=std::max(mx,q);
  int suffix=q?((b+1)*perReq-hi):0;ori.data.push_back(q?32768-suffix:0);cm.data.push_back(32768/ratio);
 }
 qs.shape.dims={B+1};ori.shape.dims={B};cm.shape.dims={B};out.data.assign(1024,0x5a5a5a5a);out.shape.dims={1024};
 SparseFlashMlaMetadataCpuKernel k;k.cuSeqlensQ_=&qs;k.sequsedOriKv_=&ori;k.sequsedCmpKv_=cmp?&cm:nullptr;k.metadata_=&out;
 k.numHeadsQ_=heads;k.numHeadsKv_=1;k.headDim_=512;k.batchSize_=B;k.maxSeqlenQ_=mx;
 k.maxSeqlenOriKv_=32768;k.maxSeqlenCmpKv_=32768/ratio;k.cmpRatio_=ratio;k.cmpTopK_=cmp?512:0;k.oriTopK_=0;
 k.oriMaskMode_=4;k.cmpMaskMode_=cmp?3:0;k.oriWinLeft_=127;k.oriWinRight_=0;
 k.layoutQ_="TND";k.layoutKv_="PA_BBND";k.hasOriKv_=true;k.hasCmpKv_=cmp;
 k.aicCoreNum_=24;k.aivCoreNum_=48;k.socVersion_="Ascend910_9382";k.isBatchConsistency_=consistency;
 if(!k.ParamsInit())std::abort();SplitResult r(24,48);if(!k.BalanceSchedule(r)||!k.GenMetadata(r))std::abort();
 bool split=false,tail=true;for(uint32_t i=0;i<r.usedCoreNum;i++)split|=out.data[i*FA_METADATA_SIZE+FA_S2_START_INDEX]!=0||out.data[i*FA_METADATA_SIZE+FA_S2_END_INDEX]!=0;
 for(int i=900;i<1024;i++)tail&=out.data[i]==0x5a5a5a5a;
 std::cout<<"{\"heads\":"<<heads<<",\"per_request_q\":"<<perReq<<",\"cp_rank\":"<<rank<<",\"compressed\":"<<cmp<<",\"ratio\":"<<ratio<<",\"batch_consistency\":"<<consistency<<",\"q_rows\":"<<qs.data.back()<<",\"mBase\":"<<k.mBaseSize_<<",\"s2Base\":"<<k.s2BaseSize_<<",\"fa_cores\":"<<r.usedCoreNum<<",\"fd_heads\":"<<r.numOfFdHead<<",\"fd_vec\":"<<r.fdRes.fdUsedVecNum<<",\"S2split\":"<<split<<",\"tail124_untouched\":"<<tail<<",\"qsl\":[";
 for(int i=0;i<=B;i++){if(i)std::cout<<",";std::cout<<qs.data[i];}std::cout<<"],\"fa_ranges\":[";
 for(uint32_t i=0;i<r.usedCoreNum;i++){if(i)std::cout<<",";std::cout<<"["<<out.data[i*FA_METADATA_SIZE+FA_BN2_START_INDEX]<<","<<out.data[i*FA_METADATA_SIZE+FA_M_START_INDEX]<<","<<out.data[i*FA_METADATA_SIZE+FA_S2_START_INDEX]<<","<<r.bN2End[i]<<","<<r.gS1End[i]<<","<<r.s2End[i]<<"]";}
 std::cout<<"],\"defined900\":[";for(int i=0;i<900;i++){if(i)std::cout<<",";std::cout<<out.data[i];}std::cout<<"]}"<<std::endl;
}
int main(){
 for(bool cmp:{false,true}){for(int r=0;r<8;r++)run(64,6,r,cmp,2,false);run(8,6,-1,cmp,2,false);}
 run(64,6,0,true,1,false);run(8,6,-1,true,1,false);
 run(64,6,0,true,2,true);run(8,6,-1,true,2,true);
 run(64,5,6,false,1,false);run(64,5,7,false,1,false);run(8,5,-1,false,1,false);
 return 0;
}
'''
 OUT.mkdir(exist_ok=False);code=prefix+meta.read_text()+'\n'+h+'\nusing namespace optiling;\nnamespace aicpu {\nuint32_t SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext&){return 0;}\n'+body+'\n}\n'+main_cpp
 (OUT/'driver.cpp').write_text(code);(OUT/'.gitignore').write_text('/driver\n')
 build=subprocess.run(['clang++','-std=c++17','-O2',str(OUT/'driver.cpp'),'-o',str(OUT/'driver')],capture_output=True,text=True);(OUT/'build.stdout').write_text(build.stdout);(OUT/'build.stderr').write_text(build.stderr)
 if build.returncode:raise RuntimeError(build.stderr)
 run=subprocess.run([str(OUT/'driver')],capture_output=True,text=True,timeout=20);(OUT/'stdout.jsonl').write_text(run.stdout);(OUT/'stderr.log').write_text(run.stderr)
 if run.returncode:raise RuntimeError(f'Fixture exited{run.returncode}: '+run.stderr)
 rows=[json.loads(l) for l in run.stdout.splitlines()];assert len(rows)==25 and all(r['tail124_untouched'] for r in rows)
 for cmp in [0,1]:
  a=[r for r in rows if r['per_request_q']==6 and r['ratio']==2 and r['compressed']==cmp and not r['batch_consistency']]
  cp=[r for r in a if r['heads']==64];glob,=[r for r in a if r['heads']==8]
  assert len(cp)==8 and sum(r['q_rows'] for r in cp)==24 and glob['q_rows']==24
  assert all(not r['S2split'] and r['fd_heads']==0 for r in cp)
  assert not glob['S2split'] and glob['fd_heads']==0
  if cmp:
   assert all(r['mBase']==64 for r in cp) and glob['mBase']==8
  else:
   assert all(r['mBase']==256 for r in cp) and glob['mBase']==256
 out=dict(status='PASS_ORIGINAL_CPU_PACKET_FIXTURES_NOT_NPU_BIT_GATE',cases=rows,source=dict(path=str(src),sha256=sha256(src.read_bytes()).hexdigest(),body_sha256=sha256(body.encode()).hexdigest(),first=text[:begin].count('\n')+1,last=text[:end].count('\n')),header_sha256=sha256(hdr.read_bytes()).hexdigest(),metadata_header_sha256=sha256(meta.read_bytes()).hexdigest(),driver_sha256=sha256(code.encode()).hexdigest(),probe_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
  limits=['Unchanged original scalar ParamsInit/BalanceSchedule/GenMetadata bodies; includes mocked/private access exposed, Compute/Prepare/ABI/device not tested.',
   'Explicit 4x6 query fixture and source CP intersection/suffix values, not captured runtime packet. Head8 global24 is new design input, not measured original path.',
   'Level3 cases are contrast only; original frozen consumer FD key not enabled by this CPU fixture. Draft20/zero-query included to expose fallback needs, not enable new headroute.',
   'All900 defined words retained. Tail124 untouched sentinel is not allocator-value or primitive1024-word equivalence. No numerical MMAD/softmax output tested.'],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,model_loads=0,new_profile=0,service_requests=0,remote_writes=[])
 (OUT/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k not in ['defined900','fa_ranges']} for r in rows],indent=2))
if __name__=='__main__':main()
