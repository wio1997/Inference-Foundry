#!/usr/bin/env python3
"""Compile original gather helpers with transfer-only CPU mocks, no arithmetic/NPU proof."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('audit/execution-search/smla-runtime-source/repo/op_kernel/arch22')
OUT=Path('audit/execution-search/smla-pair-address-cpu')

PRE=r'''
#include <algorithm>
#include <cstdint>
#include <climits>
#include <iostream>
#include <vector>
#define __aicore__
#define unlikely(x) (x)
enum class SMLA_LAYOUT {PA_BBND,BSND,TND};
struct Ptr {int64_t off=0; Ptr operator[](int64_t a)const{return {off+a};}};
struct Table {std::vector<int32_t> values; int32_t GetValue(int64_t i)const{return values.at(i);}};
struct DataCopyExtParams {uint32_t blockLen=0,blockCount=0,dstStride=0,srcStride=0;};
template<class T> struct DataCopyPadExtParams {bool pad;int a,b,c;};
std::vector<int64_t> trace;
template<class T> void DataCopyPad(Ptr dst,Ptr src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 for(uint32_t i=0;i<p.blockCount;++i){
  int64_t dest=dst.off+i*(p.blockLen+p.dstStride)/sizeof(T);
  int64_t source=src.off+i*(p.blockLen+p.srcStride)/sizeof(T);
  for(uint32_t j=0;j<p.blockLen/sizeof(T);j+=512){
   if(trace.size()<=static_cast<size_t>((dest+j)/512))trace.resize((dest+j)/512+1,-999);
   trace[(dest+j)/512]=source+j;
  }
 }
}
struct RunInfo {int64_t cmpS2IdLimit,bIdx=0,n2Idx=0,tensorCmpBOffset=0;};
struct Info {int64_t paCmpBlockSize=64,cmpMaxBlockNumPerBatch=2,cmpKvStride0=64*512,kvHeadNum=1,headDim=512,cmpSeqSize=128,sparseBlockSize=1;};
template<class SMLAT> class SMLAVectorBlock {
public:
 using KV_T=uint16_t;
 static constexpr SMLA_LAYOUT KV_LAYOUT_T=SMLA_LAYOUT::PA_BBND;
 static constexpr int INPUT2_BUFFER_OFFSET=16384;
 Info constInfo;Table cmpBlockTableGm_;Ptr kvMergUb_,cmpKvGm_;
 int64_t GetKeyGmOffset(int64_t,const RunInfo&,int64_t);
 void CopyInSingleKv(int64_t&,int64_t,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void CopyInKv(int64_t&,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
};
'''

POST=r'''
int main(){
 struct Case{const char* name;std::vector<int32_t> pages;int64_t a,b,limit;int64_t block=64;};
 std::vector<Case> cases={
  {"ascending_page_boundary",{3,11},63,64,128},
  {"reverse_page_boundary_fast_pair",{11,3},63,64,128},
  {"same_page_gap",{3,11},5,19,128},
  {"invalid_second",{3,11},63,-1,128},
  {"causal_clipped_second",{3,11},63,64,64},
  {"both_invalid",{3,11},-1,128,128},
  {"C1_reverse_page_fast_pair",{11,3},127,128,256,128},
  {"C1_reverse_page_stride_overflow_fallback",{17303,0},127,128,256,128}
 };
 std::cout<<"[";
 for(size_t ci=0;ci<cases.size();++ci){
  auto c=cases[ci];SMLAVectorBlock<int> op;op.cmpBlockTableGm_.values=c.pages;
  op.constInfo.paCmpBlockSize=c.block;op.constInfo.cmpKvStride0=c.block*512;
  RunInfo info{c.limit};int64_t mte2=0;trace.clear();op.CopyInKv(mte2,0,0,c.a,c.b,info);
  std::vector<int64_t> logical;
  for(auto key:{c.a,c.b}){auto p=op.GetKeyGmOffset(key,info,c.limit);if(p>=0)logical.push_back(p);}
  bool naive=(trace==logical);
  // A valid direct reader must follow this emitted address order, rather than logical order.
  std::cout<<(ci?",":"")<<"{\"case\":\""<<c.name<<"\",\"original_addresses\":[";
  for(size_t j=0;j<trace.size();++j)std::cout<<(j?",":"")<<trace[j];
  std::cout<<"],\"logical_addresses\":[";
  for(size_t j=0;j<logical.size();++j)std::cout<<(j?",":"")<<logical[j];
  std::cout<<"],\"naive_logical_order_matches\":"<<(naive?"true":"false")<<",\"copied_rows\":"<<mte2<<"}";
 }
 std::cout<<"]\n";
}
'''


def main():
    p=ROOT/'sparse_flash_mla_csa_block_vector.h';s=p.read_text()
    helpers=[];ranges=[]
    for name in ['GetKeyGmOffset','CopyInSingleKv','CopyInKv']:
        pat=r'template <typename SMLAT>\n__aicore__ inline [^\n]*SMLAVectorBlock<SMLAT>::'+name+r'\('
        m=re.search(pat,s);assert m,name
        start=m.start();brace=s.index('{',m.end());depth=1;end=brace+1
        while depth:
            if s[end]=='{':depth+=1
            elif s[end]=='}':depth-=1
            end+=1
        helpers.append(s[start:end]);ranges.append(dict(function=name,first=s.count('\n',0,start)+1,last=s.count('\n',0,end)+1))
    source=PRE+'\n\n'.join(helpers)+POST
    OUT.mkdir(exist_ok=True);driver=OUT/'driver.cpp';driver.write_text(source)
    build=subprocess.run(['clang++','-std=c++17','-O2',str(driver),'-o',str(OUT/'probe')],capture_output=True,text=True)
    (OUT/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
    run=subprocess.run([str(OUT/'probe')],capture_output=True,text=True,check=True)
    rows=json.loads(run.stdout);assert len(rows)==8
    assert [r['case'] for r in rows if not r['naive_logical_order_matches']]==['reverse_page_boundary_fast_pair','C1_reverse_page_fast_pair']
    assert rows[-1]['naive_logical_order_matches']
    result=dict(status='ORIGINAL_CPP_HELPERS_CPU_ADDRESS_TRACE_CONFIRMED',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_ranges=ranges,driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),cases=rows,scope='PA_BBND BF16-size elements, sparseBlockSize1, C2 cmpBlock64 and C1 cmpBlock128 with17304 allocated pages as matched typed cohort. Transfer mock only, no attention arithmetic, NPU output or performance validation.',limits=['Native ND2NZ/MTE behavior and full pair/gather retirement are not executed','C1 block128 span exceeds2GiB; the large-stride pair fallback preserves logical order, unlike the fast physical-order pair','Different original address order is not a measured attention bit counterexample'],new_NPU_work=0,whole_chain_complete=False)
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--attempt');args=ap.parse_args()
    if args.attempt:
        assert args.attempt.isalnum();OUT=Path('audit/execution-search/smla-pair-address-cpu-'+args.attempt)
    main()
