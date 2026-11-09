#!/usr/bin/env python3
"""Guarded fixed-workload prefix-ID assembly, isolated CPU native point only."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from replay_input_renderer_ids import REMOTE as REPLAY_REMOTE

INNER=r'''
from pathlib import Path
from hashlib import sha256
import sys,sysconfig,json,time,random,copy,statistics
sys.path.insert(0,sysconfig.get_path('purelib'))
import tokenizers
from tokenizers import Tokenizer
assert tokenizers.__version__=='0.22.2'
data=json.load(sys.stdin);raw=data['content'].encode();assert sha256(raw).hexdigest()=='7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404'
rows=[json.loads(s) for s in raw.decode().splitlines()];texts=[r.get('question',r.get('prompt')) for r in rows];assert len(texts)==8 and all(isinstance(t,str) for t in texts)
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/tokenizer.json');b=model.read_bytes();assert sha256(b).hexdigest()=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
meta=json.loads(b);assert meta['model'].get('dropout') is None
assert meta['normalizer']==dict(type='Sequence',normalizers=[])
first=meta['pre_tokenizer']['pretokenizers'][0];assert first==dict(type='Split',pattern=dict(Regex=r'\p{N}{1,3}'),behavior='Isolated',invert=False)
assert meta['pre_tokenizer']['pretokenizers'][-1]['type']=='ByteLevel' and not meta['pre_tokenizer']['pretokenizers'][-1]['add_prefix_space']
assert meta['post_processor']['type']=='ByteLevel'
added=meta['added_tokens'];assert not any(a.get(k) for a in added for k in ['lstrip','rstrip','single_word'])
contents=[a['content'] for a in added];partials={c[:i] for c in contents for i in range(1,len(c))};maxspecial=max(map(len,contents))
full=Tokenizer.from_file(str(model));suffix=copy.deepcopy(full);suffix.no_padding();suffix.no_truncation()
def old(text,cap=33793):
 full.enable_truncation(max_length=cap,direction='right');full.no_padding()
 return full.encode_batch([text],add_special_tokens=True)[0].ids
def safe(prefix):
 if len(prefix)<3:return False
 tail=prefix[-3:]
 if not(tail.isascii() and tail.isdecimal() and (len(prefix)==3 or prefix[-4].isascii() and not prefix[-4].isdigit())):return False
 return not any(prefix[-k:] in partials for k in range(1,min(len(prefix),maxspecial-1)+1))
def build(prefix):
 return tuple(suffix.encode_batch([prefix],add_special_tokens=False)[0].ids)
def cached(text,prefix,ids,cap=33793):
 if not isinstance(text,str) or not safe(prefix) or not text.startswith(prefix):return old(text,cap)
 # Validate full Python text first. Invalid UTF8 falls back to the original
 # full API so Unicode errors retain their original positions/messages.
 try:text.encode('utf8')
 except UnicodeEncodeError:return old(text,cap)
 out=list(ids);out.extend(suffix.encode_batch([text[len(prefix):]],add_special_tokens=False)[0].ids)
 return out[:cap] if len(out)>cap else out
def outcome(func,*args):
 try:return ('ids',func(*args))
 except Exception as e:return ('error',type(e).__name__,str(e))
import os
common=os.path.commonprefix(texts);prefix=common[:117067];assert len(common)==117286 and safe(prefix)
start=time.perf_counter_ns();pids=build(prefix);build_ms=(time.perf_counter_ns()-start)/1e6
counts=dict(cases=0,ID_elements=0,mismatches=0,fallback_cases=0)
fingerprint=sha256();fails=[]
def compare(text,prefix,ids,cap=33793):
 a=outcome(old,text,cap);bb=outcome(cached,text,prefix,ids,cap);counts['cases']+=1
 counts['fallback_cases']+=int(not isinstance(text,str) or not safe(prefix) or not text.startswith(prefix) or isinstance(text,str) and any(0xd800<=ord(c)<=0xdfff for c in text))
 if a[0]=='ids':counts['ID_elements']+=len(a[1])
 fingerprint.update(repr(a).encode('utf8','backslashreplace'))
 if a!=bb:
  counts['mismatches']+=1
  if len(fails)<8:fails.append(dict(case=counts['cases'],cap=cap,old_sha256=sha256(repr(a).encode('utf8','backslashreplace')).hexdigest(),new_sha256=sha256(repr(bb).encode('utf8','backslashreplace')).hexdigest()))
for text in texts:
 for cap in [1,17,33793]:compare(text,prefix,pids,cap)
vocab=suffix.get_vocab();rng=random.Random(20261110)
units=['hello',' 世界','かな','🙂','123456789','\n\t','é','e\u0301','<|partial','<｜Assistant｜>','�','']+contents[:20]
for i in range(1200):
 pre=''.join(rng.choice(units) for _ in range(rng.randrange(1,5)))+'!123'
 tail=''.join(rng.choice(units) for _ in range(rng.randrange(0,7)))
 if safe(pre):compare(pre+tail,pre,build(pre),[1,17,33793][i%3])
for pre in ['hello','12','!1234','<|reserved123','!123']:
 ids=build(pre)
 for tail in ['hello','456789','｜>','\ud800','a\udfffz','']:
  compare(pre+tail,pre,ids)
compare(12,prefix,pids);compare('unmatched input',prefix,pids)
naive=[]
for word in ['hello','tokenization','12345','hello world','中国人']:
 target=suffix.encode_batch([word],add_special_tokens=False)[0].ids
 for k in range(1,len(word)):
  split=suffix.encode_batch([word[:k]],add_special_tokens=False)[0].ids+suffix.encode_batch([word[k:]],add_special_tokens=False)[0].ids
  if target!=split:naive.append(dict(text=word,cut=k,whole_ids=target,split_ids=split));break
assert naive
# Prefix tuples immutable, each emitted list independent.
a=cached(texts[0],prefix,pids);bb=cached(texts[0],prefix,pids);saved=bb[0];a[0]=-1;assert bb[0]==saved and pids[0]==saved
rounds=[]
for repeat in range(9):
 arms={}
 for new in ([False,True] if repeat%2==0 else [True,False]):
  start=time.perf_counter_ns()
  output=[cached(t,prefix,pids) if new else old(t) for t in texts]
  arms['new_ms' if new else 'old_ms']=(time.perf_counter_ns()-start)/1e6
  assert all(len(ids)==32768 for ids in output)
 rounds.append(arms)
oldms=statistics.median(r['old_ms'] for r in rounds);newms=statistics.median(r['new_ms'] for r in rounds)
binary=next(Path(sysconfig.get_path('purelib')).glob('tokenizers/tokenizers*.so'));binary_sha=sha256(binary.read_bytes()).hexdigest();assert binary_sha=='fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print('PREFIX_TOKENIZER_RESULT='+json.dumps(dict(status='PASS_LIMITED_PREFIX_ID_PRIMITIVE' if counts['mismatches']==0 else 'FAIL',counts=counts,mismatches=fails,
 compared_fingerprint=fingerprint.hexdigest(),binary_sha256=binary_sha,tokenizer_JSON_sha256=sha256(b).hexdigest(),
 prefix=dict(chars=len(prefix),utf8_bytes=len(prefix.encode()),IDs=len(pids),sha256=sha256(prefix.encode()).hexdigest(),
  approximate_python_ID_tuple_bytes=sys.getsizeof(pids)+sum(sys.getsizeof(i) for i in pids),safe_numeric_ASCII3=True,proper_added_prefix_collision=False),
 shared_chars=len(common),first_prefix_materialization_ms=build_ms,paired_rounds=rounds,
 local_CPU=dict(old8_median_ms=oldms,warm_new8_median_ms=newms,warm_net_ms=oldms-newms,
  cold_once_plus8_ms=build_ms+newms,cold_net_ms=oldms-build_ms-newms),
 naive_counterexamples=naive,output_list_ownership='Independent returned lists over immutable tuple ints; mutation control PASS',
 scope='CPU native encode_batch ID assembly only. Known workload prefix chosen from observed commonprefix, not an implemented online discovery/insertion/LRU policy. Cache excludes offsets, paired/pretokenized/MM/embedding paths and other tokenizer/normalizer/postprocessor/special flags. Source pre/post/caller guards unchanged. Truncation right with cap preserved. No service/wire/model/request-speedup admission.',
 remaining=['Generic online cache prefix discovery/insertion/native Encoding offsets + byte-LRU/epoch/ownership and cold miss/invalidation cost',
 'Actual backend/tokenizer mutation and all fallback/error/wire integration; direct source replay not captured serving IDs',
 'CPU affinity and contention, real queue/GPU-critical exposure and ordinary workloads; existing worker count changes cannot be added to shared-prefix benefit'],
 CPU_tokenizer_original_loads=1,native_clones=1,model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[],torch_imported=False,torch_npu_imported=False)))
'''


def main():
    remote=REPLAY_REMOTE.replace('INPUT_RENDERER_RESULT=','PREFIX_TOKENIZER_RESULT=')
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();x['runner_sha256']=sha256(Path('tools/replay_input_renderer_ids.py').read_bytes()).hexdigest()
    Path('audit/execution-search/shared-prefix-tokenizer-point.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:x[k] for k in ['status','counts','prefix','first_prefix_materialization_ms','local_CPU','remaining']},ensure_ascii=False))


if __name__=='__main__':main()
