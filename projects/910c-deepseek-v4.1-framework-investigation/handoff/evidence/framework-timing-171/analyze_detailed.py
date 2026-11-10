"""Same-window host-idle, kernel and communication ledgers. Existing data only."""
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,csv,heapq,json,re,statistics,sys

MACRO={'prepare input','preprocess','forward','post process','sample_token','draft_token','async_state_update'}

def merge(xs):
    out=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def intersect(a,b):
    out=[];i=j=0
    while i<len(a) and j<len(b):
        x=max(a[i][0],b[j][0]);y=min(a[i][1],b[j][1])
        if y>x:out.append([x,y])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out

def ms(xs):return sum(b-a for a,b in xs)/1000
def span(e):return [float(e['ts']),float(e['ts'])+float(e['dur'])]
def num(row,k):
    try:return float(row.get(k,0))
    except (TypeError,ValueError):return 0.0

def role(row):
    if row.get('Type')!='DispatchFFNCombine':return row.get('Type',row.get('Name','unknown'))
    pieces=row.get('Input Shapes','').strip('"').split(';')
    top=pieces[3].split(',')[-1] if len(pieces)>3 else ''
    return {'6':'DFC_target_top6','3':'DFC_draft_top3'}.get(top,'DFC_unknown')

def comm(name):return name.startswith('hcom_') or 'AicpuKernel' in name

def cpu_group(name):
    n=name.removeprefix('aten::')
    if name.startswith('Event::') or 'synchronize' in n or name.startswith('gloo:'):return '同步与事件'
    if n in {'slice','select','as_strided','view','reshape','_reshape_alias','flatten','expand','expand_as','unsqueeze','squeeze','transpose','permute','detach','alias'}:return '切片与张量视图'
    if n in {'to','_to_copy','copy_','copy','pin_memory','_pin_memory','clone','contiguous'}:return '拷贝、类型转换与内存固定'
    if any(k in n for k in ['empty','zeros','ones','fill','zero_','scalar_tensor']):return '创建与初始化小张量'
    if any(k in n for k in ['index','gather','scatter','arange','range']):return '索引与缓存位置操作'
    if name.startswith('aten::'):return '长度、条件、掩码等张量运算'
    if name.startswith(('aclnn','_C_','vllm::')) or 'kernel' in name.lower():return '算子与metadata调用封装'
    return '其他已记录Host操作'

def cpu_leaf_segments(cpu):
    """Choose deepest CPU-op by shortest active range, ties deterministically.

    This partitions event wall time, not hardware CPU execution or Python self
    time. Underlying non-cpu_op library calls remain within their CPU-op.
    """
    edges=[]
    for i,e in enumerate(cpu):
        a,b=span(e);edges.extend([(a,1,i),(b,-1,i)])
    edges.sort();active=set();heap=[];out=defaultdict(list);last=None;pos=0
    while pos<len(edges):
        t=edges[pos][0]
        while heap and heap[0][2] not in active:heapq.heappop(heap)
        if last is not None and t>last and heap:
            out[cpu[heap[0][2]]['name']].append([last,t])
        while pos<len(edges) and edges[pos][0]==t:
            _,kind,i=edges[pos]
            if kind<0:active.discard(i)
            else:
                active.add(i);a,b=span(cpu[i]);heapq.heappush(heap,(b-a,-a,i))
            pos+=1
        last=t
    return {k:merge(v) for k,v in out.items()}

def analyze(path):
    path=Path(path);raw=json.load(path.open());es=raw['traceEvents'] if isinstance(raw,dict) else raw
    rank=int(re.search(r'_ep(\d+)_',str(path))[1]);xs=defaultdict(list);cpu=[];gloo=[]
    starts=sorted(span(e)[0] for e in es if e.get('ph')=='X' and e.get('name')=='MODEL_EXECUTE')
    assert len(starts)==12,(rank,len(starts));lo,hi=starts[0],starts[-1];coverage=[[lo,hi]]
    prepare=[]
    for e in es:
        if e.get('ph')!='X' or not e.get('dur'):continue
        n=e.get('name','');a,b=span(e)
        if n in ['Free','Computing','Communication(Not Overlapped)']:xs[n].append([a,b])
        if n=='prepare input':prepare.append([a,b])
        if n=='gloo:all_reduce':gloo.append([a,b])
        if e.get('cat')=='cpu_op' and e.get('pid')==e.get('tid') and n not in MACRO:cpu.append(e)
    xs={k:intersect(merge(v),coverage) for k,v in xs.items()};prep=intersect(merge(prepare),coverage)
    prep_free=intersect(xs['Free'],prep);leaves=cpu_leaf_segments(cpu);host=[];group_spans=defaultdict(list)
    for name,iv in leaves.items():
        in_p=intersect(iv,prep);in_free=intersect(iv,prep_free)
        count=sum(1 for e in cpu if e['name']==name and lo<=span(e)[0]<hi and intersect([[span(e)[0],span(e)[1]]],prep))
        host.append({'name':name,'category':cpu_group(name),'calls':count,
                     'self_host_ms_in_prepare':ms(in_p),'self_host_ms_during_free':ms(in_free)})
        group_spans[cpu_group(name)].extend(in_free)
    grouped={k:ms(merge(v)) for k,v in group_spans.items()}
    covered=sum(grouped.values());assert covered<=ms(prep_free)+1e-5
    grouped['无CPU-op范围覆盖的框架区间']=max(0,ms(prep_free)-covered)
    host.sort(key=lambda e:-e['self_host_ms_during_free'])
    # Exact same closed-cycle device sample, including all shape signatures.
    by_role=defaultdict(list)
    with (path.parent/'kernel_details.csv').open() as stream:
        for r in csv.DictReader(stream):
            a=num(r,'Start Time(us)');b=a+num(r,'Duration(us)');x=max(a,lo);y=min(b,hi)
            if y>x:r['_iv']=[x,y];by_role[role(r)].append(r)
    ops=[]
    case_keys=['Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num']
    for name,rows in by_role.items():
        iv=merge([r['_iv'] for r in rows]);wall=sum((r['_iv'][1]-r['_iv'][0]) for r in rows)/1000
        aic=sum(num(r,'aicore_time(us)') for r in rows);cases=defaultdict(list)
        for r in rows:cases[tuple(r.get(k) for k in case_keys)].append(r)
        examples=[]
        for signature,cr in cases.items():
            examples.append({'calls':len(cr),'wall_sum_ms':sum(r['_iv'][1]-r['_iv'][0] for r in cr)/1000,
                             'fields':dict(zip(case_keys,signature))})
        graph=[r['_iv'] for r in rows if str(r.get('Model ID','')) not in ['','4294967295','-1']]
        dur=[num(r,'Duration(us)') for r in rows]
        ops.append({'name':name,'kind':'communication' if comm(name) else 'operator','calls':len(rows),'union_ms':ms(iv),
                    'window_pct':ms(iv)/(hi-lo)*1000*100,'wall_sum_ms':wall,
                    'call_median_us':statistics.median(dur),'call_p90_us':sorted(dur)[int((len(dur)-1)*.9)],
                    'target_graph_union_ms':ms(merge(graph)),
                    'aic_mac_active_pct':sum(num(r,'aic_mac_time(us)') for r in rows)/aic*100 if aic else None,
                    'aic_mte2_active_pct':sum(num(r,'aic_mte2_time(us)') for r in rows)/aic*100 if aic else None,
                    'shape_cases':sorted(examples,key=lambda e:-e['wall_sum_ms'])})
    ops.sort(key=lambda r:-r['union_ms'])
    return {'ep':rank,'dp':rank//8,'trace':str(path),'csv':str(path.parent/'kernel_details.csv'),
            'closed_start_us':lo,'closed_end_us':hi,'cycles':11,'window_ms':(hi-lo)/1000,
            'device_union_ms':{k:ms(v) for k,v in xs.items()},'prepare_free_ms':ms(prep_free),
            'host_self_groups_ms':grouped,'host_ops':host,'operators':ops}

if __name__=='__main__':
    root=Path(sys.argv[1]);paths=sorted((root/'traces').rglob('trace_view.json'));assert len(paths)==16
    out={'scope':'same scoped capture; 11 closed cycles per rank; no new profiling','ranks':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(analyze,paths):
            out['ranks'].append(row);print('detailed',row['ep'],len(row['operators']),flush=True)
    (root/'detailed-performance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('DETAIL_COMPLETE',len(out['ranks']),flush=True)
