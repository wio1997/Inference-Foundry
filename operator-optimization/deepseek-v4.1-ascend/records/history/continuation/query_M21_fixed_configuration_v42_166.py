from pathlib import Path
from datetime import datetime,timezone
import json,re,shlex,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
M=P/'analysis/native-PMU-complete-v18';OUT=M/'M21-captured-fixed-parallel-spec-and-dispatch-configuration-v42.json';assert not OUT.exists()
pins={'runtime/scripts/serve.sh':'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba','runtime/scripts/serve-host-profile.sh':'2cd68992493ef97750ae3fd9832d3c301bec319f22b657cf4f07f6a8ac3f3766','runtime/logs/service.log':'a02e582afaf94f0911b0bfb62ddac9028f2ac25936e4776373fc0ebfd2e1c61e','runtime/config/serve.env':'569c372788ba565df9a116f6cded3060e2f12caac07f8c3c304515093a2b381a'}
names=set('tensor_parallel_size pipeline_parallel_size data_parallel_size prefill_context_parallel_size decode_context_parallel_size enable_expert_parallel enable_dsa_cp disable_padded_drafter_batch enable_chunked_prefill num_speculative_tokens method enforce_eager max_num_seqs max_num_batched_tokens kv_cache_dtype attention_backend enable_sparse_kv_offload sample_from_anchor dynamic_spec_config enable_flashcomm1 enable_fused_mc2 cudagraph_mode cudagraph_num_of_warmups cudagraph_capture_sizes cudagraph_copy_inputs max_cudagraph_capture_size enable_npugraph_ex enable_static_kernel enable_ascend_quant sample_from_anchor use_mega_moe'.split())
flags={'--tensor-parallel-size','--pipeline-parallel-size','--data-parallel-size','--prefill-context-parallel-size','--decode-context-parallel-size','--enable-expert-parallel','--kv-cache-dtype','--max-num-seqs','--max-num-batched-tokens','--max-model-len','--attention-backend','--quantization','--disable-hybrid-kv-cache-manager','--async-scheduling','--enforce-eager'}
jsonflags={'--additional-config','--speculative-config','--compilation-config'}
envnames={'VLLM_USE_V1','VLLM_DSV41_SAMPLE_FROM_ANCHOR','VLLM_ASCEND_ENABLE_DSA_CP','VLLM_ASCEND_ENABLE_DSV41_PACKED_CACHE','VLLM_DSV41_USE_PACKED_CACHE','VLLM_ASCEND_ENABLE_PCP'}
secret=re.compile('password|api.?key|access.?key|secret|credential|auth.?token',re.I)
def select(obj,path=''):
    rows=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if secret.search(str(k)):continue
            q=path+'.'+str(k)if path else str(k)
            if k in names:
                if isinstance(v,(str,int,float,bool,type(None))):rows.append(dict(field=q,value=v))
                elif k=='dynamic_spec_config'and isinstance(v,dict):
                    for kk,vv in v.items():
                        if not secret.search(str(kk))and isinstance(vv,(str,int,float,bool,type(None))):rows.append(dict(field=q+'.'+str(kk),value=vv))
            if isinstance(v,(dict,list)):rows+=select(v,q)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            if isinstance(v,(dict,list)):rows+=select(v,path+'['+str(i)+']')
    return rows
keypattern=re.compile(r'''(?<![A-Za-z0-9_])['"]?('''+'|'.join(sorted(names,key=len,reverse=True))+r''')['"]?\s*[:=]\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|True|False|None|null|true|false|-?\d+(?:\.\d+)?|\[[^\]\n]{0,180}\])''')
records=[]
for rel,sha in pins.items():
    p=P/rel;before=p.stat();b=p.read_bytes();after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert hashlib.sha256(b).hexdigest()==sha,(rel,hashlib.sha256(b).hexdigest());txt=b.decode(errors='replace')
    rows=[]
    if rel.endswith('.sh'):
        for ln,line in enumerate(txt.splitlines(),1):
            if line.startswith('exec ')and'vllm serve 'in line:
                args=shlex.split(line)
                for i,a in enumerate(args):
                    if a in jsonflags and i+1<len(args):rows.append(dict(line=ln,flag=a,selected=select(json.loads(args[i+1]))))
                    elif a in flags:
                        value=args[i+1]if i+1<len(args)and not args[i+1].startswith('--')else True
                        rows.append(dict(line=ln,flag=a,value=value))
    elif rel.endswith('.env'):
        for ln,line in enumerate(txt.splitlines(),1):
            line=line.strip().removeprefix('export ');key,sep,val=line.partition('=')
            if sep and key in envnames:rows.append(dict(line=ln,name=key,value=val.strip()))
    else:
        for ln,line in enumerate(txt.splitlines(),1):
            if 'non-default args:'not in line and'Initializing a V1 LLM engine'not in line:continue
            typ='EFFECTIVE_ENGINE_LOG'if'Initializing a V1 LLM engine'in line else'PARSER_NON_DEFAULT_ARGS'
            for match in keypattern.finditer(line):
                literal=match.group(2);assert not secret.search(literal)
                # Occurrences stay unmerged: main/draft nestedsettings canrepeat.
                rows.append(dict(line=ln,record_type=typ,field=match.group(1),literal=literal,char_offset=match.start()))
    records.append(dict(path=str(p),sha256=sha,bytes=len(b),original_stat_unchanged=True,selected_fields=rows,full_text_not_emitted=True))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),records=records,scope='Exactsavedcaptureruntimefiles/SHA,onlytechnicalwhitelist;no currentprocess/environment/modelobjectquery.',limits=['Absenceofafieldisnotitsdefaultvalue. Repeated/nestedmain-versus-draftfieldsremainseparatewithoffsets.','Parser/CLIlogversuseffectiveEngine startupareseparateauthorities;notnewhealth/idleorloadedPythonobjectproof.'],no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
