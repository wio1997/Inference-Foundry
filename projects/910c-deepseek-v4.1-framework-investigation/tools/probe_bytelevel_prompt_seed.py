#!/usr/bin/env python3
"""Isolated CPU tokenizer primitive point. No service requests or remote files.

Original constructor is retained even for the guarded empty-seed alternative.
Only bytes/None/error fingerprints and timing summaries leave the process.
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,sys,sysconfig,time,random,statistics
root=Path(sysconfig.get_path('purelib'));sys.path.insert(0,str(root))
import tokenizers
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream,ByteLevel
assert tokenizers.__version__=='0.22.2'
inputdata=json.load(sys.stdin);ip=Path(inputdata['path']);iraw=inputdata['content'].encode()
assert sha256(iraw).hexdigest()=='7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404'
rows=[json.loads(l) for l in iraw.decode().splitlines()];assert len(rows)==8
tp=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/tokenizer.json')
raw=tp.read_bytes();assert sha256(raw).hexdigest()=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
meta=json.loads(raw);assert meta['decoder']['type']=='ByteLevel'
t=Tokenizer.from_file(str(tp));assert isinstance(t.decoder,ByteLevel)
vocab=t.get_vocab(with_added_tokens=True);vids=sorted(set(vocab.values()))
special={r['id'] for r in meta['added_tokens'] if r.get('special')}

def guard(ids,skip):
    # Actual per-ID native decode preserves added-vocab/skip behaviour. The
    # immutable ByteLevel object and retained constructor are explicit gates.
    if not isinstance(t.decoder,ByteLevel):return False
    for token in reversed(ids):
        text=t.decode([token],skip_special_tokens=skip)
        if text:return ord(text[-1])<128
    return False

def construct(ids,skip,new=False):
    stream=DecodeStream(ids=ids,skip_special_tokens=skip)
    if new and guard(ids,skip):stream=DecodeStream(skip_special_tokens=skip)
    return stream

def record(stream,token):
    try:
        s=stream.step(t,token)
        return ('none',) if s is None else ('bytes',s.encode('utf8'))
    except Exception as e:return ('error',type(e).__name__,str(e))

counts=dict(sequences=0,steps=0,guarded_sequences=0,fallback_sequences=0,mismatches=0)
digest=sha256();fail=[]
def compare(ids,seq,skip):
    a,b=construct(ids,skip),construct(ids,skip,True);g=guard(ids,skip)
    counts['sequences']+=1;counts['guarded_sequences' if g else 'fallback_sequences']+=1
    for k,z in enumerate(seq):
        aa,bb=record(a,z),record(b,z);counts['steps']+=1
        digest.update(repr(aa).encode())
        if aa!=bb:
            counts['mismatches']+=1
            if len(fail)<6:fail.append(dict(prefix_id_sha256=sha256(json.dumps(ids).encode()).hexdigest(),step=k,skip=skip,guard=g,
              old_sha256=sha256(repr(aa).encode()).hexdigest(),new_sha256=sha256(repr(bb).encode()).hexdigest()))

ascii_ids=t.encode('safe ASCII boundary.\n',add_special_tokens=False).ids
assert ascii_ids and guard(ascii_ids,True)
single={}
for v in vids:
    token=t.id_to_token(v)
    if token is not None and len(token)==1:single[token]=v
# GPT2 byte-to-unicode alphabet, mirrored solely to select adversarial byte IDs.
bs=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xad))+list(range(0xae,0x100));cs=bs[:];n=0
for b in range(256):
    if b not in bs:bs.append(b);cs.append(256+n);n+=1
bytechar=dict(zip(bs,map(chr,cs)));byteid={b:single[c] for b,c in bytechar.items() if c in single}
assert len(byteid)==256
boundary_sequences=[[],[byteid[x] for x in b'abc'],[byteid[0xe4],byteid[0xb8],byteid[0xad]],
 [byteid[0xf0],byteid[0x9f],byteid[0x98],byteid[0x80]],
 [byteid[0xff],byteid[0x80],byteid[0x61]],sorted(special)[:6],
 [-1,byteid[0x61],2**32,byteid[0x62],None,'invalid',byteid[0x63]],
 [[byteid[0xe4],byteid[0xb8]],byteid[0xad],[],[byteid[0x61],byteid[0x62]]]]
prefixes=[[],ascii_ids,ascii_ids+[byteid[0xe4]],ascii_ids+[byteid[0xe4],byteid[0xb8]],
 ascii_ids+[byteid[0xff]],ascii_ids+[byteid[0xe4],byteid[0x61]],sorted(special)[:3],
 ascii_ids+sorted(special)[:3],ascii_ids+[byteid[0xef],byteid[0xbf],byteid[0xbd]]]
for skip in [False,True]:
    for ids in prefixes:
        for seq in boundary_sequences:compare(ids,seq,skip)
    # Every vocabulary ID is tested as a first continuation and then followed
    # by ASCII+partial/malformed bytes: equality includes subsequent state.
    for v in vids:compare(ascii_ids,[v,byteid[0x61],byteid[0xe4],byteid[0xb8],byteid[0xad]],skip)
rng=random.Random(20261110)
for i in range(256):
    prefix=[rng.choice(vids) for _ in range(rng.randrange(1,64))]+[byteid[0x61]]
    seq=[rng.choice(vids) for _ in range(32)]+boundary_sequences[2]
    compare(prefix,seq,bool(i%2))

real=[];performance=[]
for i,row in enumerate(rows):
    prompt=row.get('question',row.get('prompt'));assert isinstance(prompt,(str,list))
    ids=t.encode(prompt,add_special_tokens=True).ids if isinstance(prompt,str) else prompt
    rec=dict(index=i,payload_type=type(prompt).__name__,ids_count=len(ids),ids_sha256=sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
      guard_skip_true=guard(ids,True),guard_skip_false=guard(ids,False),max_tokens=row.get('max_tokens',1024))
    real.append(rec)
    for skip in [False,True]:
        for seq in boundary_sequences:compare(ids,seq,skip)
    # Paired alternating local CPU primitive calls, not a service/model result.
    # Includes original constructor, native guard and candidate extra stream.
    old=[];new=[]
    for repeat in range(9):
        for alt in ([False,True] if repeat%2==0 else [True,False]):
            start=time.perf_counter_ns();s=construct(ids,True,alt);r=record(s,byteid[0x61]);elapsed=(time.perf_counter_ns()-start)/1e6
            (new if alt else old).append(elapsed)
    performance.append(dict(index=i,old_ms=old,new_ms=new,old_median_ms=statistics.median(old),new_median_ms=statistics.median(new),
      median_net_saved_ms=statistics.median(old)-statistics.median(new)))

# Execute original vLLM classes/methods, avoiding the vLLM/torch module imports.
import ast,itertools
from abc import ABC,abstractmethod
from types import SimpleNamespace as S
sp=Path('/vllm-workspace/vllm/vllm/v1/engine/detokenizer.py');src=sp.read_bytes()
assert sha256(src).hexdigest()=='213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93'
tree=ast.parse(src);chosen=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['IncrementalDetokenizer','BaseIncrementalDetokenizer','FastIncrementalDetokenizer','check_stop_strings']]
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*chosen],type_ignores=[])
env=dict(ABC=ABC,abstractmethod=abstractmethod,tokenizers=tokenizers,sys=sys,
 logger=S(exception=lambda *args:None,warning=lambda *args:None),INVALID_PREFIX_ERR_MSG='Invalid prefix encountered')
exec(compile(ast.fix_missing_locations(mod),str(sp),'exec'),env);Original=env['FastIncrementalDetokenizer']
class Candidate(Original):
    def __init__(self,backend,request):
        super().__init__(backend,request)
        if guard(request.prompt_token_ids,self.skip_special_tokens):self.stream=DecodeStream(skip_special_tokens=self.skip_special_tokens)
wrapper=dict(cases=0,updates=0,mismatches=0)
seq=[byteid[b] for b in b'abSTOPcd']+sorted(special)[:2]+boundary_sequences[2]+[byteid[0x61]]
real0=rows[0].get('question',rows[0].get('prompt'));realids=t.encode(real0,add_special_tokens=True).ids if isinstance(real0,str) else real0
for prefix,skip,spaces,include,stops,min_tokens,delta,width in itertools.product(
 [ascii_ids,ascii_ids+[byteid[0xe4]],realids],[False,True],[False,True],[False,True],[[],['STOP'],['abS','STOP','TOP']], [0,3],[False,True],[1,5]):
    req=S(request_id='cpu-fixture',prompt_token_ids=prefix,sampling_params=S(skip_special_tokens=skip,spaces_between_special_tokens=spaces,
      stop=stops,min_tokens=min_tokens,include_stop_str_in_output=include))
    a,b=Original(S(_tokenizer=t),req),Candidate(S(_tokenizer=t),req);wrapper['cases']+=1
    for k in range(0,len(seq),width):
        piece=seq[k:k+width];finished=k+width>=len(seq)
        aa=a.update(piece,finished);bb=b.update(piece,finished)
        va=(aa,a.get_next_output_text(finished,delta).encode(),a.output_text.encode(),a.token_ids,a._last_output_text_offset)
        vb=(bb,b.get_next_output_text(finished,delta).encode(),b.output_text.encode(),b.token_ids,b._last_output_text_offset)
        wrapper['updates']+=1
        if va!=vb:wrapper['mismatches']+=1
        if aa or bb:break

# Counterexample: removing the guard changes the existing initial UTF8 state.
p=[byteid[0xe4]];a=construct(p,True);b=DecodeStream(skip_special_tokens=True)
unguarded=[record(a,z)!=record(b,z) for z in [byteid[0xb8],byteid[0xad],byteid[0x61]]]
assert any(unguarded)
construct_errors=[]
for ids in [[-1],[2**32],['x'],None]:
    # None is a legal constructor value but request prompt IDs are a list.
    if ids is None:continue
    ee=[]
    for alt in [False,True]:
        try:construct(ids,True,alt);ee.append(None)
        except Exception as e:ee.append((type(e).__name__,str(e)))
    assert ee[0]==ee[1];construct_errors.append(dict(input_type=type(ids[0]).__name__,error_type=ee[0][0] if ee[0] else None))
binary=next(root.glob('tokenizers/tokenizers*.so'));binary_bytes=binary.read_bytes()
out=dict(status='PASS_LIMITED_CPU_PRIMITIVE' if counts['mismatches']==wrapper['mismatches']==0 else 'FAIL',
  candidate='Retain original DecodeStream(ids=prompt) constructor; for immutable ByteLevel and final effective token endingASCII use new empty seed, otherwise keep original stream.',
  tokenizer_json_sha256=sha256(raw).hexdigest(),tokenizers_version=tokenizers.__version__,
  binary=dict(path=str(binary),bytes=len(binary_bytes),sha256=sha256(binary_bytes).hexdigest()),
  input=dict(path=str(ip),sha256=sha256(iraw).hexdigest(),add_special_tokens=True,
    scope='Direct tokenizers reconstruction, not captured server-renderer IDs; count32768 alone is not identity proof'),
  counts=counts,compared_output_fingerprint=digest.hexdigest(),mismatches=fail,vocabulary_ids=len(vids),
  byte_ids=256,skip_special_tokens_cases=[False,True],boundary_prefixes=len(prefixes),
  constructor_errors=construct_errors,unguarded_UTF8_counterexample_mismatch_steps=unguarded,original_vllm_wrapper=wrapper,
  same_input_reconstruction=real,paired_CPU_primitive=performance,
  timing_scope='9 pairs per input, alternating arms in isolated CPU process after CPU tokenizer load; includes old ctor in both arms, new guard and extra ctor in candidate. Not a benchmark of service/request/NPU or decoded model tokens.',
  remaining=['ByteLevel concat proof/guard with original constructor and all legal primitive inputs, tokenizer immutability and full vLLM spacing/stop/SSE wrapper tests',
    'Actual renderer ID hash/flags binding and full output-token trace missing; direct CPU reconstruction is not actual served ID capture',
    'Real endpoint exposure/new host costs and six ordinary paths; no replacement deployed or NPU bit acceptance'],
  CPU_tokenizer_vocabulary_loads=1,model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[],service_changes=[])
print(json.dumps(out))
'''


REMOTE=r'''
from hashlib import sha256
import json,subprocess,sys
from pathlib import Path
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running']
assert item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inputs=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/benchmark-20261028/workload/frozen/measure1')
ip,=list(inputs.rglob('*.requests.jsonl'));raw=ip.read_bytes()
assert sha256(raw).hexdigest()=='7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404'
payload=json.dumps(dict(path=str(ip),content=raw.decode()))
r=subprocess.run(['docker','exec','-i',name,'python3','-B','-I','-S','-c',sys.argv[1]],input=payload,capture_output=True,text=True)
if r.returncode:
 print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout)
x.update(container_id=item['Id'],image=item['Image']);print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-4000:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    p=Path('audit/execution-search/bytelevel-prompt-seed-point.json');p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=x['status'],counts=x['counts'],real_input_guard=[r['guard_skip_true'] for r in x['same_input_reconstruction']],
      old_sum_median_ms=sum(r['old_median_ms'] for r in x['paired_CPU_primitive']),new_sum_median_ms=sum(r['new_median_ms'] for r in x['paired_CPU_primitive']),remote_writes=[])))


if __name__=='__main__':main()
