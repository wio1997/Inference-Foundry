#!/usr/bin/env python3
"""Consume the existing full Encoding to populate a shared immutable prefix."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from probe_shared_prefix_tokenizer import INNER as PREFIX_INNER
from replay_input_renderer_ids import REMOTE as REPLAY_REMOTE

INNER=PREFIX_INNER.split('import os\ncommon=')[0]+r'''
import os
common=os.path.commonprefix(texts);prefix=common[:117067];cut=len(prefix);assert safe(prefix)
def extract(encoding,ids,text,position):
 # encode_batch uses character offsets. Never use the UTF8-byte length here.
 lo=0;hi=len(ids)
 while lo<hi:
  mid=(lo+hi)//2;span=encoding.token_to_chars(mid)
  if span is None:return None
  if span[1]<=position:lo=mid+1
  else:hi=mid
 if lo==0 or encoding.token_to_chars(lo-1)[1]!=position:return None
 if lo<len(ids) and encoding.token_to_chars(lo)[0]<position:return None
 key=text[:position];hash(key)
 return key,tuple(ids[:lo])
full.enable_truncation(max_length=33793,direction='right');full.no_padding()
encodings=full.encode_batch(texts,add_special_tokens=True);ids=[e.ids for e in encodings]
reference=build(prefix);checks=[]
for i,(text,e,ii) in enumerate(zip(texts,encodings,ids)):
 got=extract(e,ii,text,cut);assert got is not None and got[0]==prefix and got[1]==reference
 assert cached(text,got[0],got[1])==ii
 checks.append(dict(index=i,IDs=len(ii),prefix_IDs=len(got[1]),last_prefix_char_span=e.token_to_chars(len(got[1])-1),first_suffix_char_span=e.token_to_chars(len(got[1]))))
# Repeated producer handoff only, no extra prefix encoding in its timing.
times=[]
for i in range(31):
 start=time.perf_counter_ns();got=extract(encodings[0],ids[0],texts[0],cut);elapsed=(time.perf_counter_ns()-start)/1e6;times.append(elapsed)
 assert got[1]==reference
wrongpos=len(prefix.encode());wrong=extract(encodings[0],ids[0],texts[0],wrongpos)
byte_misuse=dict(char_cut=cut,wrong_byte_cut=wrongpos,byte_cut_rejected=wrong is None,
  wrong_prefix_IDs=None if wrong is None else len(wrong[1]),would_equal_true_prefix=False if wrong is None else wrong[1]==reference)
assert wrong is None or wrong[1]!=reference
full.enable_truncation(max_length=17,direction='right');small=full.encode_batch([texts[0]],add_special_tokens=True)[0]
assert extract(small,small.ids,texts[0],cut) is None
# The prototype misses on request0 and shares only request1..7. Include cache
# extraction/key hash/ID tuple in cold path, no separately reencoded prefix.
rounds=[]
for repeat in range(9):
 arms={}
 for new in ([False,True] if repeat%2==0 else [True,False]):
  full.enable_truncation(max_length=33793,direction='right');start=time.perf_counter_ns()
  if new:
   e=full.encode_batch([texts[0]],add_special_tokens=True)[0];first=e.ids;key,pids=extract(e,first,texts[0],cut)
   values=[first]+[cached(text,key,pids) for text in texts[1:]]
  else:values=[old(text) for text in texts]
  arms['cold_handoff8_ms' if new else 'old8_ms']=(time.perf_counter_ns()-start)/1e6
  assert values==ids
 rounds.append(arms)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print('PREFIX_HANDOFF_RESULT='+json.dumps(dict(status='PASS_LIMITED_ENCODING_PREFIX_HANDOFF',request_checks=checks,
 original_full_IDs_equal=True,cap17_incomplete_prefix_not_inserted=True,byte_offset_counterexample=byte_misuse,
 extraction_ms=times,median_insertion_ms=statistics.median(times),max_insertion_ms=max(times),
 paired_rounds=rounds,local_CPU=dict(old8_median_ms=statistics.median(r['old8_ms'] for r in rounds),
  cold_handoff8_median_ms=statistics.median(r['cold_handoff8_ms'] for r in rounds)),
 scope='Original full Encoding/IDs for first miss retained; prefix key hash and tuple from token_to_chars binary search,7 subsequent known-prefix hits. No online prefix-discovery/insertion scheduling/LRU implementation; predetermined cut and actual source caller integration remain pending. Native objects/IDs only, not captured serving process.',
 CPU_tokenizer_original_loads=1,native_clones=1,torch_imported=False,torch_npu_imported=False,model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])))
'''


def main():
    remote=REPLAY_REMOTE.replace('INPUT_RENDERER_RESULT=','PREFIX_HANDOFF_RESULT=')
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();x['prefix_probe_sha256']=sha256(Path('tools/probe_shared_prefix_tokenizer.py').read_bytes()).hexdigest()
    Path('audit/execution-search/prefix-encoding-handoff-point.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:x[k] for k in ['status','median_insertion_ms','max_insertion_ms','local_CPU','byte_offset_counterexample']},ensure_ascii=False))


if __name__=='__main__':main()
