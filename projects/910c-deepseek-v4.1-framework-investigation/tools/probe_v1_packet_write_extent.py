#!/usr/bin/env python3
"""Run exact GenMetadata bodies with output sentinels on host C++ only."""
from pathlib import Path
from hashlib import sha256
import json,re,subprocess
ROOT=Path('audit/execution-search')
def body(text,start):
    a=text.index(start);b=text.index('{',a);depth=1;i=b+1
    while depth:
        depth += (text[i]=='{')-(text[i]=='}');i+=1
    return text[a:i]
def main():
    d=json.loads((ROOT/'v1-packet-dependencies.json').read_text());sources={s['runtime_path']:s for s in d['sources']}
    def source(suf):
        s=next(s for k,s in sources.items() if k.endswith(suf));b=Path(s['local_path']).read_bytes();assert sha256(b).hexdigest()==s['runtime_sha256'];return s,b.decode()
    generated=[];pins=[]
    for label,stem,cls,structure in [('SMLA','sparse_flash_mla','SparseFlashMlaMetadataCpuKernel','SmlaMetadata'),('QLI','quant_lightning_indexer_v2','QuantLightningIndexerV2MetadataCpuKernel','QliV2Metadata')]:
        hsuf=('/sparse_flash_mla_kernel_metadata.h' if label=='SMLA' else '/quant_lightning_indexer_v2_metadata.h')
        h,header=source('/op_kernel'+hsuf);c,cpp=source('/op_kernel_aicpu/'+stem+'_metadata_aicpu.cpp');fn=body(cpp,'bool '+cls+'::GenMetadata(')
        assert '*metadataPtr = {};' in fn
        header=re.sub(r'^#include .*\n','',header,flags=re.M)
        mock='''
struct FakeTensor {uint32_t* p; void* GetData(){return p;}};
struct FD {uint32_t fdUsedVecNum=0;std::vector<uint32_t> fdIdx,fdBN2Idx,fdMIdx,fdWorkspaceIdx,fdS2SplitNum,fdMStart,fdMNum; FD():fdIdx(72),fdBN2Idx(72),fdMIdx(72),fdWorkspaceIdx(72),fdS2SplitNum(72),fdMStart(72),fdMNum(72){}};
struct SplitResult {uint32_t usedCoreNum=0,maxS2LoopNum=0;std::vector<uint32_t>bN2End,gS1End,s2End,firstFdDataWorkspaceIdx;FD fdRes;SplitResult():bN2End(36),gS1End(36),s2End(36),firstFdDataWorkspaceIdx(36){}};
'''
        klass='class '+cls+' {public:FakeTensor* metadata_;uint32_t aicCoreNum_=24,aivCoreNum_=48;bool isSplitG_=false;bool GenMetadata(SplitResult&);};\n'
        main='''
void test(){
 for(auto cores: {24u,32u,36u})for(auto used:{0u,1u}){
  std::array<uint32_t,1024>a,b;a.fill(0x11111111u);b.fill(0x22222222u);FakeTensor ta{a.data()},tb{b.data()};
  SplitResult s;s.usedCoreNum=used;s.fdRes.fdUsedVecNum=used;
  CLASS k;k.aicCoreNum_=cores;k.aivCoreNum_=2*cores;k.metadata_=&ta;k.GenMetadata(s);k.metadata_=&tb;k.GenMetadata(s);
  constexpr size_t n=sizeof(optiling::detail::STRUCT)/sizeof(uint32_t);bool eq=true,tail=true;
  for(size_t i=0;i<n;i++)eq &= a[i]==b[i];
  for(size_t i=n;i<1024;i++)tail &= a[i]==0x11111111u && b[i]==0x22222222u;
  if(!eq||!tail||n>=1024)std::abort();
  std::cout << "{\\"operator\\":\\"LABEL\\",\\"aic\\":"<<cores<<",\\"used\\":"<<used<<",\\"struct_words\\":"<<n<<",\\"tail_words\\":"<<1024-n<<",\\"struct_equal\\":true,\\"tail_sentinels_unchanged\\":true,\\"full_packet_equal\\":false}\\n";
 }
}
'''.replace('CLASS',cls).replace('STRUCT',structure).replace('LABEL',label)
        generated.append('namespace '+label+' {\n'+header+'\nusing namespace optiling;\n'+mock+klass+fn+main+'}\n');pins += [dict(runtime_path=x['runtime_path'],sha256=x['runtime_sha256']) for x in [h,c]]
    code='#include <cstdint>\n#include <array>\n#include <vector>\n#include <iostream>\n#include <cstdlib>\n'+''.join(generated)+'int main(){SMLA::test();QLI::test();}\n'
    out=ROOT/'v1-packet-write-extent-cpu';out.mkdir(exist_ok=True);driver=out/'driver.cpp';driver.write_text(code);binary=out/'driver'
    compile=subprocess.run(['/usr/bin/clang++','-std=c++17','-O0','-Wall','-Wextra',str(driver),'-o',str(binary)],capture_output=True,text=True,timeout=40)
    (out/'compile.log').write_text(compile.stdout+compile.stderr);assert compile.returncode==0,compile.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10);(out/'run.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
    records=[json.loads(x) for x in run.stdout.splitlines()];assert len(records)==12
    result=dict(status='PASS_ORIGINAL_GENMETADATA_WRITE_EXTENT_ONLY',records=records,sources=pins,driver_sha256=sha256(driver.read_bytes()).hexdigest(),
        qualification='Original complete GenMetadata bodies and packet headers, host C++ with explicit mock Tensor/SplitResult. Scalar serializer extent controls, not full valid schedules, CANN execution or bitwise native replacement admission. Only the defined struct is overwritten by these functions; framework/runtime outside these functions may initialize padding and is not inferred here.',
        consequence='Source-level pointer-bearing packet-field hypothesis not supported by these integer-only structs/serializers. Do not infer all1024words deterministic or silently ignore496/640padding bytes. Strict full primitive byte gate stays open; retain original packets and verify actual loaded output initialization/allocation before replacement.',
        new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'v1-packet-write-extent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(cases=len(records),SMLA_tail_bytes=496,QLI_tail_bytes=640,new_native_admissions=0)))
if __name__=='__main__':main()
