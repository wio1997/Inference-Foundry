import ctypes,json,pathlib,statistics,threading,time
from concurrent.futures import ThreadPoolExecutor
lib=ctypes.CDLL(None,use_errno=True)
assert lib.prctl(29,1,0,0,0)==0
class TS(ctypes.Structure):
 _fields_=[('sec',ctypes.c_long),('nsec',ctypes.c_long)]
barrier=threading.Barrier(33)
def probe(i):
 barrier.wait(timeout=10)
 target=time.perf_counter_ns()+int((12+i*.005)*1e9)
 if i%2:
  ts=TS(*divmod(target,10**9));rc=lib.clock_nanosleep(time.CLOCK_MONOTONIC,1,ctypes.byref(ts),None);assert rc==0
 else:time.sleep((target-time.perf_counter_ns())/1e9)
 return {'mode':'absolute_clock_nanosleep' if i%2 else 'relative_python_sleep','target_ns':target,'lag_ms':(time.perf_counter_ns()-target)/1e6}
with ThreadPoolExecutor(max_workers=32) as pool:
 f=[pool.submit(probe,i) for i in range(32)];barrier.wait(timeout=10);rows=[x.result() for x in f]
out={'completed':True,'scope':'CPU-only timing control outside formal model timing; identical12sec lookahead,32prestarted threads. No model requests.','clock':str(time.get_clock_info('perf_counter')),'timerslack_ns':pathlib.Path('/proc/self/timerslack_ns').read_text().strip(),'rows':rows}
for mode in ['absolute_clock_nanosleep','relative_python_sleep']:
 a=[r['lag_ms'] for r in rows if r['mode']==mode];out[mode]={'min_ms':min(a),'median_ms':statistics.median(a),'max_ms':max(a)}
print(json.dumps(out,indent=2))
