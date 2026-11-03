from pathlib import Path
import json

N = Path('/Users/wio/work/166-vllm-analysis-continuation-20261030')
O = N / 'profile-first-20261031'
A = '/vllm-workspace/vllm-ascend/vllm_ascend/'
V = '/vllm-workspace/vllm/vllm/'
specs = [
    (A+'models/deepseek_v41/dspark.py', 'fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601', 'dspark_fields'),
    (A+'models/deepseek_v41/model.py', '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91', 'decoder_tail_and_projection_fields'),
    (A+'ops/vocab_parallel_embedding.py', '4e06e8bc22d693659818a117790ee604cda041b11d66902b49d756714d61ca0f', 'embedding_fields'),
    (V+'model_executor/layers/vocab_parallel_embedding.py', 'd42574ac79bd7d7847548784d2272c7c40cbd93fe6cdadb15e8c2db79d2074d7', 'embedding_fields'),
    (V+'distributed/communication_op.py', None, 'allreduce_wrapper'),
    (A+'attention/dsa_v41.py', '3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd', 'actual_attention_forward'),
]
code = r'''from pathlib import Path
import ast,hashlib,json
specs=SPECS;out=[]
for path,expected,mode in specs:
    p=Path(path);b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
    if expected is not None:assert sha==expected,(path,sha)
    ls=b.decode().splitlines();t=ast.parse(b.decode());par={c:z for z in ast.walk(t)for c in ast.iter_child_nodes(z)};stmts={};functions=[];imports=[]
    def context(z):
        owner=None;guards=[];q=z
        while q in par:
            v=par[q]
            if isinstance(v,ast.If):guards.append(dict(line=v.lineno,test=ast.unparse(v.test),arm='BODY'if q in v.body else 'ORELSE'if q in v.orelse else 'TEST'))
            if isinstance(v,ast.FunctionDef):owner=v.name;break
            q=v
        classes=[c.name for c in ast.walk(t)if isinstance(c,ast.ClassDef)and c.lineno<=z.lineno<=c.end_lineno]
        return owner,guards,classes
    def extract(start,end):return[str(j+1)+': '+ls[j]for j in range(start-1,end)]
    attrs=({'embed_tokens','hc_mult','use_sequence_parallel_moe'}if mode=='dspark_fields'else {'wq_a','compress_ratio','is_draft_layer'}if mode=='decoder_tail_and_projection_fields'else {'tp_rank','tp_size','tp_group'}if mode=='embedding_fields'else set())
    for z in ast.walk(t):
        if not isinstance(z,(ast.Assign,ast.AnnAssign)):continue
        targets=z.targets if isinstance(z,ast.Assign)else[z.target]
        names={q.attr for x in targets for q in ast.walk(x)if isinstance(q,ast.Attribute)}
        if mode=='embedding_fields':names.update(q.id for x in targets for q in ast.walk(x)if isinstance(q,ast.Name)and q.id in attrs)
        if not names&attrs:continue
        owner,guards,classes=context(z);assert z.end_lineno-z.lineno<50
        stmts[z.lineno]=dict(start=z.lineno,end=z.end_lineno,attributes=sorted(names),owner=owner,classes=classes,guards=guards,source=extract(z.lineno,z.end_lineno))
    if mode in ['decoder_tail_and_projection_fields','actual_attention_forward','allreduce_wrapper']:
        fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and ((mode=='decoder_tail_and_projection_fields'and f.lineno<=905<=f.end_lineno)or(mode=='actual_attention_forward'and f.lineno<=701<=f.end_lineno)or(mode=='allreduce_wrapper'and f.name=='tensor_model_parallel_all_reduce'))]
        assert len(fs)==1,(path,mode)
        for f in fs:
            start=900 if mode=='decoder_tail_and_projection_fields'else f.lineno
            assert f.end_lineno-start<100
            functions.append(dict(name=f.name,start=f.lineno,end=f.end_lineno,selected_start=start,source=extract(start,f.end_lineno)))
    helpers={'get_tp_group','get_tensor_model_parallel_world_size','get_tensor_model_parallel_rank','tensor_model_parallel_all_reduce'}
    for z in t.body:
        if isinstance(z,ast.ImportFrom)and any(a.name in helpers for a in z.names):imports.append(dict(line=z.lineno,source=extract(z.lineno,z.end_lineno)))
    out.append(dict(path=path,sha256=sha,previous_SHA_pinned=expected is not None,mode=mode,class_headers=[dict(name=c.name,line=c.lineno,bases=[ast.unparse(a)for a in c.bases])for c in t.body if isinstance(c,ast.ClassDef)]if mode=='embedding_fields'else [],statements=list(stmts.values()),functions=functions,imports=imports))
assert sum(sum(len(s['source'])for s in f['statements'])+sum(len(s['source'])for s in f['functions'])for f in out)<450
print(json.dumps(dict(files=out,scope='Only profile-raised decoder tail, actual DSA forward, embedding TP fields/AllReduce wrapper and DSpark entry/projection fields; prior HC/SP/DSA preprocessing contracts reused.',no_model_or_torch_import=True)))
'''.replace('SPECS', repr(specs))
outer = """from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M19-selected-decoder-tail-embedding-TP-owner-and-attention-output-contract-v35.json';assert not OUT.exists()
code=CODE
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
""".replace('CODE', repr(code))
compile(outer, 'prepared-remote-source-reader', 'exec')
(N/'read_M19_decoder_and_embedding_owners_v35_166.py').write_text(outer)
gate = json.loads((O/'PROFILE-CODE-QUESTION-GATE-v35.json').read_text())
gate.update(new_source_read_allowed=True,
    source_question='104 complete patterns,10/20 global vs2/3 physical local rows,19cache-only patterns/57KV and207liveQ. Same opaque TP8 group perDB does not alone prove embedding reduction/SP ownership; exact group source, decoder tail fanout and noQ output obligations need point contracts.',
    allowed_files=[path+': '+mode for path,sha,mode in specs],
    reuse_rule='Do not reread v7 SP shard/mask, v16 DSA350-389/443-492, v31 collapse, v32 head, v34 context/ReplicatedKV. No whole-model/SDK audit. Previously unsnapshotted communication wrapper will be read exactly and SHA-recorded, without claiming historical binary attestation.')
(O/'PROFILE-CODE-QUESTION-GATE-v35.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
(N/'zcode-M19-source-v35-task-20261031.txt').write_text('只用一次exec运行准备脚本，不修改、不扩展、不重试、不额外验证、不写memory。只读profile定位的decoder尾部、DSA小forward、embedding TP字段和小allreduce wrapper、DSpark入口/投影字段；不读整模型/SDK、不import模型/torch、不做NPU/服务/健康操作。失败保留真实rc停止。\npython3 '+str(N/'remote_run.py')+' 166 '+str(N/'read_M19_decoder_and_embedding_owners_v35_166.py')+' M19-source166-v35\n只报告真实rc/receipt。\n')
print('Prepared6 point-source files; remote script syntax valid')
