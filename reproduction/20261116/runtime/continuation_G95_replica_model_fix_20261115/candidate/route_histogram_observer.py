"""Finite diagnostic observer of an ALREADY reduced expert histogram.

No routing, mask, capacity, output, extra collective, group or weight mutation.
Only EP rank0 copies metadata after the original max().item() has synchronized.
Never use this environment for a scored benchmark.
"""
from pathlib import Path
import datetime,json,os,time,traceback

ROOT=Path(os.environ['SGLANG_G95_ROUTE_CAPTURE_DIR'])
SESSION='G95_route_load_20261115'
PLACEMENT=json.loads((Path('/task')/'continuation_G95_replica_model_fix_20261115/placement.json').read_text())['physical_to_logical_map']
_state=None
_disabled=False


def write(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def observe(hist,capacity,layer_id,global_tokens,group):
    global _state,_disabled
    if _disabled or group.rank()!=0 or not (ROOT/'ARMED').is_file():
        return
    try:
        marker=json.loads((ROOT/'ARMED').read_text())
        assert marker=={'session':SESSION,'max_windows':2,'layers':40}
        assert isinstance(layer_id,int) and 0<=layer_id<40,layer_id
        if _state is None:
            if layer_id!=0:return
            _state={'window':0,'expected_layer':0,'records':0,'session':SESSION}
        assert layer_id==_state['expected_layer'],(_state,layer_id)
        assert len(global_tokens)==4 and all(isinstance(n,int) and n>=0 for n in global_tokens)
        assert hist.shape==(400,) and str(hist.dtype)=='torch.int32'
        # Read-only copy: the producer is the original histogram all-reduce.
        values=hist.cpu().tolist();assert len(values)==400 and min(values)>=0
        owners=[sum(values[i:i+25]) for i in range(0,400,25)]
        assert sum(values)==6*sum(global_tokens),(sum(values),global_tokens)
        assert max(1,max(owners))==capacity
        logical=[0]*384
        for p,n in enumerate(values):logical[PLACEMENT[layer_id][p]]+=n
        value={'logical384':logical,'session':SESSION,'window':_state['window'],'layer_id':layer_id,
               'at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'captured_epoch_s':time.time(),'captured_monotonic_s':time.monotonic(),
               'global_DP_rows':list(global_tokens),'total_active_routes':sum(values),
               'received_capacity':capacity,'owners16':owners,'experts400':values,
               'pid_in_container':os.getpid(),'EP_rank':0,'diagnostic_only':True}
        write('window_%d_layer_%02d.json'%(_state['window'],layer_id),value)
        _state['records']+=1;_state['expected_layer']+=1
        if layer_id==39:
            _state['window']+=1;_state['expected_layer']=0
        if _state['window']==2:
            _disabled=True
        write('state.json',{**_state,'status':'disabled_complete' if _disabled else 'capturing',
                            'disabled':_disabled,'no_new_collective':True,
                            'histogram_or_output_mutation':False})
    except Exception:
        # A diagnostic failure invalidates collection, never changes model math.
        _disabled=True
        write('state.json',{'status':'disabled_diagnostic_error','disabled':True,
                            'error':traceback.format_exc(),'partial_state':_state})
