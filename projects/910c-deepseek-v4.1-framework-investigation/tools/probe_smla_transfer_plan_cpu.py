#!/usr/bin/env python3
"""Compare an independent address-plan emitter with unchanged original V0 helpers."""
from pathlib import Path
from hashlib import sha256
import json,re,subprocess,random,zlib,argparse

ROOT=Path('audit/execution-search')
parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='a2');args=parser.parse_args()
OUT=ROOT/f'smla-transfer-plan-cpu-{args.attempt}'
SRC=ROOT/'smla-runtime-source/repo/op_kernel/arch22/sparse_flash_mla_csa_block_vector.h'

PRE=r'''
#include <algorithm>
#include <cstdint>
#include <climits>
#include <iostream>
#include <map>
#include <vector>
#define __aicore__
#define unlikely(x) (x)
namespace AscendC{enum class HardEvent{MTE3_MTE2,MTE2_MTE3};}
template<AscendC::HardEvent E>void SetFlag(int64_t){}
template<AscendC::HardEvent E>void WaitFlag(int64_t){}
int sub=0;int GetSubBlockIdx(){return sub;}
template<class T> T CeilDiv(T a,T b){return (a+b-1)/b;}
template<class T,class U> T SMLAAlign(T a,U b){return CeilDiv(a,T(b))*T(b);}
enum class SMLA_LAYOUT{PA_BBND,BSND,TND};
enum Tag{KV,UB,PANEL};
struct Ptr{Tag tag;int64_t off=0;Ptr operator[](int64_t x)const{return {tag,off+x};}};
struct Table{std::vector<int32_t> values;int32_t GetValue(int64_t i)const{return values.at(i);}};
struct DataCopyExtParams{uint32_t blockLen=0,blockCount=0,dstStride=0,srcStride=0;};
template<class T>struct DataCopyPadExtParams{bool pad;int a,b,c;};
std::map<int64_t,int64_t> ub[2],panel;
template<class T>void DataCopyPad(Ptr dst,Ptr src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 for(uint32_t b=0;b<p.blockCount;b++)for(uint32_t j=0;j<p.blockLen/sizeof(T);j+=512){
  auto d=dst.off+b*(p.blockLen+p.dstStride)/sizeof(T)+j;
  auto s=src.off+b*(p.blockLen+p.srcStride)/sizeof(T)+j;
  ub[sub][d/512]=s;
 }
}
void DataCopyPad(Ptr dst,Ptr src,DataCopyExtParams p){
 for(uint32_t b=0;b<p.blockCount;b++)for(uint32_t j=0;j<p.blockLen/2;j+=512){
  auto d=dst.off+b*(p.blockLen+p.dstStride)/2+j;
  auto s=src.off+b*(p.blockLen+p.srcStride)/2+j;
  panel[d/512]=ub[sub].at(s/512);
 }
}
struct RunInfo{int64_t cmpS2IdLimit=0,bIdx=0,n2Idx=0,tensorCmpBOffset=0,v0S2DealSize=512,v0S2Start=0,topKBaseOffset=0,cmpLoop=0;};
struct Info{int64_t paCmpBlockSize=64,cmpMaxBlockNumPerBatch=0,cmpKvStride0=64*512,kvHeadNum=1,headDim=512,cmpSeqSize=0,sparseBlockSize=1,sparseBlockCount=512;};
template<class SMLAT>class SMLAVectorBlock{public:
 using KV_T=uint16_t;static constexpr SMLA_LAYOUT KV_LAYOUT_T=SMLA_LAYOUT::PA_BBND;
 static constexpr int INPUT2_BUFFER_OFFSET=16384,SYNC_INPUT_BUF2_FLAG=4,MERGE_CACHE_GM_BUF_NUM=3;
 Info constInfo;Table cmpBlockTableGm_,topkGm_;Ptr kvMergUb_{UB},cmpKvGm_{KV},kvMergeGm_{PANEL};uint32_t mergeMte3Idx=0;
 void GetRealS2Idx(int64_t,int64_t&,int64_t,const RunInfo&);
 int64_t GetKeyGmOffset(int64_t,const RunInfo&,int64_t);
 void CopyInSingleKv(int64_t&,int64_t,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void CopyInKv(int64_t&,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void CopyOutMrgeResult(int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void ProcessVec0L(const RunInfo&);
};
'''

POST=r'''
int main(){int count;std::cin>>count;std::cout<<"[";
 for(int t=0;t<count;t++){
  int block,limit,seed,ring,np;std::cin>>block>>limit>>seed>>ring>>np;
  std::vector<int32_t> pages(np),ids(512);for(auto &v:pages)std::cin>>v;for(auto &v:ids)std::cin>>v;
  panel.clear();ub[0].clear();ub[1].clear();
  for(sub=0;sub<2;sub++){
   SMLAVectorBlock<int> op;op.constInfo.paCmpBlockSize=block;op.constInfo.cmpKvStride0=block*512;
   op.constInfo.cmpMaxBlockNumPerBatch=np;op.cmpBlockTableGm_.values=pages;op.topkGm_.values=ids;op.mergeMte3Idx=seed;
   RunInfo r;r.cmpS2IdLimit=limit;r.cmpLoop=ring;op.ProcessVec0L(r);
  }
  std::cout<<(t?",":"")<<"[";bool first=true;for(auto [d,s]:panel){if(!first)std::cout<<",";first=false;std::cout<<"["<<(d-(ring%3)*512)<<","<<s<<"]";}std::cout<<"]";
 }
 std::cout<<"]\n";
}
'''

def extract(name,text):
 m=re.search(r'template <typename SMLAT>\n__aicore__ inline [^\n]*SMLAVectorBlock<SMLAT>::'+name+r'\(',text);assert m,name
 start=m.start();i=text.index('{',m.end());depth=1;j=i+1
 while depth:
  depth+=(text[j]=='{')-(text[j]=='}');j+=1
 return text[start:j],dict(function=name,first=text.count('\n',0,start)+1,last=text.count('\n',0,j)+1)

def emit(ids,pages,block,limit):
 result={};mode_counts={'fast':0,'fallback':0}
 def address(i):return -1 if i<0 or i>=limit else pages[i//block]*block*512+(i%block)*512
 for start,end in [(0,256),(256,512)]:
  dest=start
  for i in range(start,end,2):
   a,b=ids[i:i+2]
   if a<0:break
   x,y=address(a),address(b)
   if x<0 and y<0:continue
   stride=(abs(x-y)-512)*2
   fallback=stride>=2**31-1 or stride<0 or a+1>=limit or b+1>=limit
   mode_counts['fallback' if fallback else 'fast']+=1
   chosen=[p for p in [x,y] if p>=0]
   if not fallback:chosen.sort()
   for p in chosen:result[dest]=p;dest+=1
 return result,mode_counts

def cover(addresses):
 # Exact minimum positive affine-run cover, bounded to one128-row L1 subtile.
 out=[]
 for base in range(0,len(addresses),128):
  a=addresses[base:base+128];n=len(a);dp=[(10**9,None)]*(n+1);dp[n]=(0,None)
  for i in range(n-1,-1,-1):
   choices=[i+1]
   if i+1<n:
    step=a[i+1]-a[i]
    if 0<step<=65535:
     j=i+2;choices.append(j)
     while j<n and a[j]-a[j-1]==step:j+=1;choices.append(j)
   j=min(choices,key=lambda j:1+dp[j][0]);dp[i]=(1+dp[j][0],j)
  i=0
  while i<n:
   j=dp[i][1];out.append(dict(dest=base+i,rows=j-i,src=a[i],stride=a[i+1]-a[i] if j-i>1 else 512));i=j
 return out

def fixtures():
 cases=[];rng=random.Random(352201)
 for block in [64,128]:
  limit=32768//(128//block);np=(limit+block-1)//block
  maps={'ascending':list(range(1,np+1)),'reverse':list(range(np,0,-1)),'wide':[(i*137)%17304 for i in range(np)]}
  for K in [0,1,2,3,15,16,17,63,64,65,127,128,129,255,256,257,511,512]:
   sets={'contiguous':list(range(63,63+K)),'spaced':[i*limit//max(K,1) for i in range(K)],'random':sorted(rng.sample(range(limit),K))}
   for mapname,pages in maps.items():
    for distribution,ids in sets.items():cases.append(dict(name=f'B{block}/K{K}/{mapname}/{distribution}',block=block,limit=limit,pages=pages,ids=ids+[-1]*(512-K),K=K,seed=0,ring=0))
 # Positive out-of-range is outside prepared-ID guarantee but exercises strict fallback guards.
 for K in [17,257,512]:
  c=dict(cases[-1]);c.update(name=f'guard_clipped_positive/K{K}',K=K,ids=list(range(K))+[-1]*(512-K));c['ids'][3]=c['limit']+1;cases.append(c)
 for seed in [1,2]:
  for ring in [1,2,3]:
   c=dict(cases[-4]);c.update(name=f'pingpong{seed}/ring{ring}',seed=seed,ring=ring);cases.append(c)
 return cases

def main():
 OUT.mkdir(exist_ok=False);text=SRC.read_text();bodies=[];spans=[]
 for f in ['GetRealS2Idx','GetKeyGmOffset','CopyInSingleKv','CopyInKv','CopyOutMrgeResult','ProcessVec0L']:
  b,r=extract(f,text);bodies.append(b);spans.append(r)
 code=PRE+'\n\n'.join(bodies)+POST;(OUT/'driver.cpp').write_text(code)
 build=subprocess.run(['clang++','-std=c++17','-O2',str(OUT/'driver.cpp'),'-o',str(OUT/'driver')],capture_output=True,text=True);(OUT/'build.log').write_text(build.stdout+build.stderr)
 if build.returncode:raise RuntimeError(build.stderr)
 cases=fixtures();parts=[str(len(cases))]
 for c in cases:parts.extend([f"{c['block']} {c['limit']} {c['seed']} {c['ring']} {len(c['pages'])}",' '.join(map(str,c['pages'])),' '.join(map(str,c['ids']))])
 run=subprocess.run([str(OUT/'driver')],input='\n'.join(parts)+'\n',capture_output=True,text=True,timeout=30);assert run.returncode==0,run.stderr;traces=json.loads(run.stdout);assert len(traces)==len(cases)
 rows=[];bad_naive=0;fallbacks=0;checked=0
 for c,trace in zip(cases,traces):
  expected,modes=emit(c['ids'],c['pages'],c['block'],c['limit']);actual=dict(trace);assert expected==actual,c['name'];checked+=len(actual)
  eligible=all(i in actual for i in range(c['K']))
  naive=[c['pages'][i//c['block']]*c['block']*512+(i%c['block'])*512 for i in c['ids'][:c['K']] if 0<=i<c['limit']]
  if list(actual.values())!=naive:bad_naive+=1
  rec={k:c[k] for k in ['name','K','block','seed','ring']};rec.update(eligible=eligible,mode_counts=modes,defined_rows=len(actual))
  if eligible:
   addresses=[actual[i] for i in range(c['K'])];groups=cover(addresses)
   for g in groups:
    assert 0<=g['stride']<=65535 and g['rows']<=128
    for r in range(g['rows']):assert g['src']+r*g['stride']==addresses[g['dest']+r]
   # QK: two D256 loads. PV: four D128 loads, each on original128-K subtiles.
   rec.update(affine_groups=len(groups),new_QK_PV_instructions=6*len(groups),old_dense_QK_PV_instructions=6*((c['K']+127)//128),largest_group=max((g['rows'] for g in groups),default=0))
  else:fallbacks+=1
  rows.append(rec)
 raw=json.dumps(dict(cases=cases,traces=traces),separators=(',',':')).encode();(OUT/'address-traces.json.zlib').write_bytes(zlib.compress(raw,9))
 result=dict(status='PASS_CPU_ORIGINAL_V0_VS_INDEPENDENT_TRANSFER_PLAN_NOT_NPU_BIT_GATE',source_sha256=sha256(SRC.read_bytes()).hexdigest(),original_function_ranges=spans,driver_sha256=sha256(code.encode()).hexdigest(),cases=len(cases),defined_row_addresses_compared=checked,naive_logical_order_mismatching_cases=bad_naive,guarded_fallback_cases=fallbacks,results=rows,trace_archive=dict(path=str(OUT/'address-traces.json.zlib'),raw_sha256=sha256(raw).hexdigest(),compressed_sha256=sha256((OUT/'address-traces.json.zlib').read_bytes()).hexdigest()),scope=['Original6 V0 functions compiled unchanged, including two AIV halves, UB chunks, pair sorting/fallback and GM ring offsets; transfer and events mocked.','Independent Python plan reproduces original defined destination-row source addresses; candidate falls back when active prefix has an unwritten old panel slot.','Positive affine groups obey2201 uint16 source stride and128-row QK/PV subtile boundaries. Checks imply identical active element source addresses for arbitrary BF16 bits, not actual ND2NZ padded lanes/LoadData/MMAD outputs or concurrency.','Synthetic contiguous/spaced/random rows and page maps; not measured model K/page statistics or DMA speed. Existing nativeQLI fixtures are evaluated separately.'],new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[])
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','cases','defined_row_addresses_compared','naive_logical_order_mismatching_cases','guarded_fallback_cases']}))

if __name__=='__main__':main()
