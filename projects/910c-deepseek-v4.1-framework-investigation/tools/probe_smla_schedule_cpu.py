#!/usr/bin/env python3
"""Compile unchanged scalar scheduling bodies with a small CPU tensor ABI fixture."""
from pathlib import Path
from hashlib import sha256
import json,subprocess

ROOT=Path('audit/execution-search')
OUT=ROOT/'smla-schedule-cpu-a1'

def main():
 src=ROOT/'smla-runtime-source/metadata-repo/sparse_flash_mla_metadata_aicpu.cpp';hdr=src.with_suffix('.h');meta=ROOT/'smla-runtime-source/repo/op_kernel/sparse_flash_mla_kernel_metadata.h'
 s=src.read_text();begin=s.index('int32_t SparseFlashMlaMetadataCpuKernel::GetSumOfQuerySeq()');end=s.index('\nnamespace {\nstatic const char *kernelType');body=s[begin:end]
 h='\n'.join(l for l in hdr.read_text().splitlines() if not any(x in l for x in ['"cpu_context.h"','"cpu_kernel.h"','"cpu_tensor.h"','"../../sparse_flash_mla/','"../../common/']))
 # Access-only rewrite; all scalar function bodies below remain original bytes.
 h=h.replace('private:', 'public:')
 prefix=r'''
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#define KERNEL_LOG_ERROR(...) do {} while(0)
namespace aicpu {
struct TensorShape { std::vector<int64_t> dims; int64_t GetDimSize(size_t i) {return dims.at(i);} };
struct Tensor { std::vector<int32_t> data; TensorShape shape; void* GetData(){return data.empty()?nullptr:data.data();} int64_t NumElements(){return data.size();} TensorShape* GetTensorShape(){return &shape;} };
struct CpuKernelContext{};
struct CpuKernel {virtual uint32_t Compute(CpuKernelContext&)=0;virtual ~CpuKernel()=default;};
}
'''
 main_cpp=r'''
using namespace aicpu;
struct Result {std::vector<int32_t> out;uint32_t used,fdHeads,fdVec;bool split,tailKept;};
Result run(const char* name,int q,bool batch,bool cmp,int ratio,bool empty=false) {
 SparseFlashMlaMetadataCpuKernel k;
 Tensor qs{{0,q},{{2}}},ori{{empty?0:32768},{{1}}},cm{{empty?0:32768/ratio},{{1}}},out;
 out.data.assign(1024,0x5a5a5a5a);out.shape.dims={1024};
 k.cuSeqlensQ_=&qs;k.sequsedOriKv_=&ori;k.sequsedCmpKv_=cmp?&cm:nullptr;k.metadata_=&out;
 k.numHeadsQ_=64;k.numHeadsKv_=1;k.headDim_=512;k.batchSize_=1;k.maxSeqlenQ_=q;
 k.maxSeqlenOriKv_=32768;k.maxSeqlenCmpKv_=32768/ratio;k.cmpRatio_=ratio;
 k.cmpTopK_=cmp?512:0;k.oriTopK_=0;k.oriMaskMode_=4;k.cmpMaskMode_=cmp?3:0;
 k.layoutQ_="TND";k.layoutKv_="PA_BBND";k.hasOriKv_=true;k.hasCmpKv_=cmp;
 k.aicCoreNum_=24;k.aivCoreNum_=48;k.socVersion_="Ascend910_9382";k.isBatchConsistency_=batch;
 if(!k.ParamsInit())std::abort();SplitResult r(24,48);
 if(!k.BalanceSchedule(r)||!k.GenMetadata(r))std::abort();
 bool split=false;for(uint32_t i=0;i<r.usedCoreNum;++i)split|=r.s2End[i]!=0;
 bool tail=true;for(int i=900;i<1024;i++)tail&=out.data[i]==0x5a5a5a5a;
 std::cout<<"{\"case\":\""<<name<<"\",\"q\":"<<q<<",\"batch_consistency\":"<<batch<<",\"ratio\":"<<ratio<<",\"used_cores\":"<<r.usedCoreNum<<",\"fd_heads\":"<<r.numOfFdHead<<",\"fd_vec\":"<<r.fdRes.fdUsedVecNum<<",\"S2_boundary_nonzero\":"<<split<<",\"tail124_sentinel_untouched\":"<<tail<<",\"endpoints\":[";
 for(uint32_t i=0;i<r.usedCoreNum;i++){if(i)std::cout<<",";std::cout<<"["<<r.bN2End[i]<<","<<r.gS1End[i]<<","<<r.s2End[i]<<"]";}
 std::cout<<"]}"<<std::endl;return {out.data,r.usedCoreNum,r.numOfFdHead,r.fdRes.fdUsedVecNum,split,tail};
}
int main(){
 auto c2=run("C2_q3_default",3,false,true,2);auto c1=run("C1_q3_default",3,false,true,1);
 auto fd=run("C2_q3_level3",3,true,true,2);auto large=run("C2_q24_default",24,false,true,2);
 auto largefd=run("C2_q24_level3",24,true,true,2);auto zero=run("emptyKV_q3_default",3,false,true,2,true);
 auto swa=run("denseSWA_q3_default",3,false,false,1);
 bool defined_equal=std::equal(c2.out.begin(),c2.out.begin()+900,c1.out.begin());
 std::cout<<"{\"checks\":true,\"C1_C2_defined900_equal\":"<<defined_equal<<",\"default_noFD\":"<<(c2.fdHeads==0&&!c2.split)<<",\"level3_crossS2_with_FD\":"<<(fd.fdHeads>0&&fd.split)<<"}"<<std::endl;
 return !(defined_equal&&c2.fdHeads==0&&!c2.split&&fd.fdHeads>0&&fd.split&&c2.tailKept&&fd.tailKept&&zero.used==1);
}
'''
 OUT.mkdir(exist_ok=False)
 code=prefix+meta.read_text()+'\n'+h+'\nusing namespace optiling;\nnamespace aicpu {\nuint32_t SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext&){return 0;}\n'+body+'\n}\n'+main_cpp
 (OUT/'driver.cpp').write_text(code)
 build=subprocess.run(['clang++','-std=c++17','-O2',str(OUT/'driver.cpp'),'-o',str(OUT/'driver')],capture_output=True,text=True)
 (OUT/'build.stdout').write_text(build.stdout);(OUT/'build.stderr').write_text(build.stderr)
 if build.returncode:raise RuntimeError(build.stderr)
 run=subprocess.run([str(OUT/'driver')],capture_output=True,text=True,timeout=20);(OUT/'stdout.jsonl').write_text(run.stdout);(OUT/'stderr.log').write_text(run.stderr)
 rows=[json.loads(l) for l in run.stdout.splitlines()];assert run.returncode==0
 result=dict(status='PASS_CPU_SCALAR_SCHEDULE_FIXTURE_NOT_NPU_OR_PRIMITIVE_GATE',returncode=run.returncode,source=dict(path=str(src),sha256=sha256(src.read_bytes()).hexdigest(),extracted_body_sha256=sha256(body.encode()).hexdigest(),first_line=s[:begin].count('\n')+1,last_line=s[:end].count('\n')),header_sha256=sha256(hdr.read_bytes()).hexdigest(),metadata_header_sha256=sha256(meta.read_bytes()).hexdigest(),driver_sha256=sha256(code.encode()).hexdigest(),cases=rows,limits=['Original scalar methods GetSumOfQuerySeq through GenMetadata unmodified; vendor includes mocked and private access exposed.','ParamsInit, BalanceSchedule, GenMetadata invoked directly with explicit prechecked attrs/tensors; Compute/Prepare/ParamsCheck entry and actual CANN attrs/ABI not validated.','C1/C2 equality holds only for these fully capped fixtures; not general alias/caching or1024-output primitive bit acceptance.','Tail124 untouched relative to fixture sentinel; no claim actual NPU allocator initial contents.','Level3 split is an algorithm/source witness, not measured live level3 fault or new native bit comparison.'],new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[])
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
