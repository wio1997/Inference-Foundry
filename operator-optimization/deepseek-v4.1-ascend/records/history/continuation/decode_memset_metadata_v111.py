"""CPU-only evidence decoder for one pinned legacy MemSet zero-FLOAT32 subdomain.

No device pointer is ever read. Packet extraction is a checked layout hypothesis
until a controlled native dump validates it;it is not historical model evidence.
"""
import struct

FORMAT='<4i3q2i3q6i'
FIELDS=['select_key','need_core_num','full_elems','full_bursts','front_elems','front_loops','front_part',
    'front_tail_bursts','front_tail_repeats','last_elems','last_loops','last_part',
    'last_tail_bursts','last_tail_repeats','core_num','gap_bytes','init_dtype','init_value_raw']

def decode_record(raw):
    if len(raw)!=96:raise ValueError('One source-declared record is96bytes,not a write extent')
    r=dict(zip(FIELDS,struct.unpack(FORMAT,raw)))
    if r['init_dtype']!=0 or r['init_value_raw']!=0:raise ValueError('Outside defaultFLOAT32zero subdomain')
    if not 0<=r['need_core_num']<=r['core_num']<=48 or r['core_num']==0:raise ValueError('Unbound core counts')
    if any(r[k]<0 for k in FIELDS if k not in ['select_key']):raise ValueError('Negative source-loop field')
    return r

def write_spans(r,padded_tail):
    """Relative byte spans of source store submissions;not necessary HBM traffic."""
    K=r['need_core_num'];spans=[]
    if K*(max(r['front_loops'],r['last_loops'])+1)>32768:raise ValueError('Enumeration budget exceeded')
    def put(core,start,n,kind):
        if not 0<=start<2**63 or not 0<=n<=2**32 or start+n>=2**63:raise ValueError('Invalid span domain')
        if n:spans.append(dict(core=core,begin=start,end=start+n,kind=kind))
    for core in range(K):
        last=core==K-1;prefix='last'if last else'front';base=core*r['front_elems']*4
        for j in range(r[prefix+'_loops']):put(core,base+j*r['full_elems']*4,r['full_bursts']*32,'full')
        bursts=r[prefix+'_tail_bursts']
        if bursts:
            size=bursts*32-(r['gap_bytes']if last and padded_tail else 0)
            if size<0:raise ValueError('Negative padded tail size')
            put(core,base+r[prefix+'_part']*4,size,'tail')
    merged=[]
    for s in sorted(spans,key=lambda s:(s['begin'],s['end'])):
        if merged and s['begin']<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],s['end'])
        else:merged.append([s['begin'],s['end']])
    return dict(spans=spans,submitted_store_value_bytes=sum(s['end']-s['begin']for s in spans),
        union=merged,unique_written_value_bytes=sum(b-a for a,b in merged),
        source_loop_spans_only_not_native_tiling_validity_or_physical_traffic=True)

def inline_flowtable(words,args_address,args_size,metadata):
    """Read only an already-rendered argument packet;never dereference its addresses."""
    N=metadata['compileInfo']['vars']['workspace_num'];nptr=len(metadata['parameters'])
    dfx=metadata['compileInfo']['dfx_size']
    if not 1<=N<=16 or nptr!=N+2 or metadata['opParaSize']!=96*N+dfx or dfx!=8:raise ValueError('Unmatched installed metadata')
    if len(words)*8!=args_size or args_size!=8*nptr+metadata['opParaSize']:raise ValueError('Argument-packet packing unproved')
    raw=b''.join(int(w).to_bytes(8,'little',signed=False)for w in words)
    offsets={w-args_address for w in words[:nptr]if args_address+8*nptr<=w and w+96*N<=args_address+args_size and (w-args_address)%8==0}
    if len(offsets)!=1:raise ValueError('No unique bounded inline flowtable pointer')
    offset=offsets.pop();records=[decode_record(raw[offset+96*i:offset+96*(i+1)])for i in range(N)]
    return dict(records=records,flowtable_offset=offset,source_workspace_address_words=words[:N],
        compiler_input_order_and_runtime_match_still_require_evidence=True,device_memory_dereferenced=False)
