"""Stage-container held fence and strict NPU PID/HBM evidence; no host file writes."""
import os
import re
import selectors
import subprocess
import sys
import time

EXPECTED_BOARDS=set(range(8))
EXPECTED_CHIPS={(board,chip) for board in EXPECTED_BOARDS for chip in (0,1)}
HOST_WORK_ROOT='/data/tiankuan/wio/main/'

# This entire program executes inside the already-owned stage container.
HOLDER_CODE=r'''
import fcntl,os,sys,time
path=sys.argv[1];timeout=float(sys.argv[2]);os.makedirs(path,exist_ok=True)
fd=os.open(os.path.join(path,'owner-fence.lock'),os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
end=time.monotonic()+timeout
while True:
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:
  if time.monotonic()>=end:raise TimeoutError('owner fence held')
  time.sleep(.2)
print('READY',flush=True)
sys.stdin.read(1)
'''

def acquire_owner_fence(directory,timeout=110,_launcher=None):
    if _launcher is None:
        if not directory.startswith(HOST_WORK_ROOT):raise ValueError('fence path outside task workspace')
        container_dir='/work/'+directory[len(HOST_WORK_ROOT):]
        args=['docker','exec','-i',STAGE,'python3','-u','-c',HOLDER_CODE,container_dir,str(timeout)]
        child=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    else:child=_launcher(directory,timeout)
    sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ)
    try:
        events=sel.select(timeout+5)
        if not events:child.terminate();raise TimeoutError('stage fence holder did not report readiness')
        if child.stdout.readline().strip()!='READY' or child.poll() is not None:raise RuntimeError('stage fence holder failed')
        return child
    finally:sel.close()

def ensure_owner_fence(holder):
    if holder.poll() is not None:raise RuntimeError('stage fence holder exited; no owner operation is safe')

def parse_npu_processes(text):
    headers=[i for i,line in enumerate(text.splitlines()) if 'Process id' in line and 'NPU' in line and 'Chip' in line]
    if len(headers)!=1:raise ValueError('missing or ambiguous NPU process table header')
    lines=text.splitlines();rows=lines[headers[0]+1:]
    if not rows or not re.fullmatch(r'\+[=+\-]+\+',rows[-1].strip()):raise ValueError('missing process table footer')
    pids=set();covered=set();seen=False;records=[]
    for line in rows:
        line=line.strip()
        if not line or line.startswith('+'):continue
        if not line.startswith('|') or not line.endswith('|'):raise ValueError('malformed process table row')
        cols=[x.strip() for x in line.strip('|').split('|')]
        if len(cols)==1:
            m=re.fullmatch(r'No running processes found in NPU\s+(\d+)',cols[0],re.I)
            if not m:raise ValueError('unknown process table message')
            board=int(m.group(1));covered.add(board);seen=True;continue
        if len(cols)!=4:raise ValueError('unexpected process table column count')
        m=re.fullmatch(r'(\d+)\s+(\d+)',cols[0])
        if not m or not cols[1].isdigit() or not cols[3].isdigit():raise ValueError('malformed NPU process row')
        board,chip=map(int,m.groups());pid=int(cols[1]);mem=int(cols[3])
        if board not in EXPECTED_BOARDS or chip not in (0,1) or pid<=0 or mem<0:raise ValueError('out of range NPU row')
        pids.add(pid);records.append((board,chip,pid,mem));covered.add(board);seen=True
    if not seen or covered!=EXPECTED_BOARDS:raise ValueError('incomplete NPU board process coverage: '+repr(sorted(covered)))
    return pids

def parse_npu_process_records(text):
    parse_npu_processes(text)
    rows=text.split('Process id',1)[1].splitlines();out=[]
    for line in rows:
        if not line.strip().startswith('|'):continue
        cols=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(cols)!=4:continue
        m=re.fullmatch(r'(\d+)\s+(\d+)',cols[0])
        if m and cols[1].isdigit() and cols[3].isdigit():out.append((int(m.group(1)),int(m.group(2)),int(cols[1]),int(cols[3])))
    return out

def parse_npu_hbm(text):
    head=text.split('Process id',1)[0];current=None;chips={}
    if 'HBM-Usage(MB)' not in head:raise ValueError('missing HBM header')
    for line in head.splitlines():
        if not line.lstrip().startswith('|'):continue
        cols=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(cols)!=3:continue
        card=re.match(r'^(\d+)\s+Ascend\S*$',cols[0])
        if card:current=int(card.group(1));continue
        chip=re.fullmatch(r'(\d+)\s+(\d+)',cols[0])
        if not chip:continue
        if current is None or not re.fullmatch(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]',cols[1]):raise ValueError('malformed chip identity')
        mem=re.search(r'(\d+)\s*/\s*(\d+)\s*$',cols[2])
        if not mem:raise ValueError('missing HBM reading')
        key=(current,int(chip.group(1)));used,total=map(int,mem.groups())
        if key not in EXPECTED_CHIPS or key in chips or total<=0 or used>total:raise ValueError('malformed HBM chip')
        chips[key]=(used,total)
    if set(chips)!=EXPECTED_CHIPS:raise ValueError('incomplete HBM chip coverage')
    return chips

def hardware_signature(text):
    version=re.search(r'npu-smi\s+(\S+)\s+Version:\s*(\S+)',text)
    if not version or version.group(1)!=version.group(2):raise ValueError('missing driver version')
    parse_npu_hbm(text);head=text.split('Process id',1)[0];current=None;chips={}
    for line in head.splitlines():
        if not line.lstrip().startswith('|'):continue
        cols=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(cols)!=3:continue
        card=re.match(r'^(\d+)\s+Ascend\S*$',cols[0])
        if card:current=int(card.group(1));continue
        chip=re.fullmatch(r'(\d+)\s+(\d+)',cols[0])
        if chip and current is not None:
            key=f'{current}:{int(chip.group(1))}';chips[key]={'physical_id':int(chip.group(2)),'pci':cols[1]}
    if len(chips)!=16:raise ValueError('incomplete hardware signature')
    return {'npu_smi_version':version.group(1),'chips':chips}

def build_empirical_reference(current,historical1,historical2,own):
    import hashlib
    sig=hardware_signature(current)
    if sig!=hardware_signature(historical1) or sig!=hardware_signature(historical2):raise ValueError('historical hardware/npu-smi signature mismatch')
    pids=parse_npu_processes(current)
    if not pids or not pids.issubset(own):raise ValueError('current NPU PID attribution incomplete')
    if parse_npu_processes(historical1) or parse_npu_processes(historical2):raise ValueError('historical reference has visible NPU processes')
    hist=[parse_npu_hbm(historical1),parse_npu_hbm(historical2)];cur=parse_npu_hbm(current);used={}
    for board,chip,pid,mem in parse_npu_process_records(current):
        if pid not in own:raise ValueError('non-owned process in reference')
        used[(board,chip)]=used.get((board,chip),0)+mem
    if set(used)!=EXPECTED_CHIPS:raise ValueError('not all 16 chips have attributable baseline processes')
    limits={};residual={};max_spread=0
    for key,(now,total) in cur.items():
        diff=now-used[key]
        if diff<0 or hist[0][key][1]!=total or hist[1][key][1]!=total:raise ValueError('inconsistent memory accounting')
        tag=f'{key[0]}:{key[1]}';residual[tag]=diff
        values=[hist[0][key][0],hist[1][key][0],diff]
        limits[tag]=max(values)
        spread=max(values)-min(values)
        max_spread=max(max_spread,spread)
    limits={key:value+max_spread for key,value in limits.items()}
    return {'method':'observed-per-chip-maximum-plus-largest-observed-spread','observed_spread_margin_mb':max_spread,'hardware_signature':sig,'envelope_mb':limits,'baseline_residual_mb':residual,
            'history_sha256':[hashlib.sha256(x.encode()).hexdigest() for x in (historical1,historical2)],
            'limitations':'empirical observations; invisible users, boot changes and process accounting uncertainty remain'}

def verified_release(text,reference=None):
    """An empirical operational check, not proof no unseen user exists."""
    pids=parse_npu_processes(text);hbm=parse_npu_hbm(text)
    if pids:return {'pids':pids,'hbm':hbm,'satisfies_policy':False,'reason':'PID_PRESENT'}
    if not reference:return {'pids':pids,'hbm':hbm,'satisfies_policy':False,'reason':'NO_EMPIRICAL_REFERENCE'}
    if hardware_signature(text)!=reference.get('hardware_signature'):raise ValueError('release hardware signature mismatch')
    limits=reference.get('envelope_mb',{})
    if set(limits)!={f'{a}:{b}' for a,b in EXPECTED_CHIPS}:raise ValueError('incomplete empirical reference')
    if any(type(v)!=int or v<0 for v in limits.values()):raise ValueError('invalid empirical reference')
    excess={f'{a}:{b}':used-limits[f'{a}:{b}'] for (a,b),(used,total) in hbm.items() if used>limits[f'{a}:{b}']}
    return {'pids':pids,'hbm':hbm,'satisfies_policy':not excess,'reason':'WITHIN_EMPIRICAL_ENVELOPE' if not excess else 'UNATTRIBUTED_HBM_ABOVE_REFERENCE','excess_mb':excess}

def wait_for_empirical_release(snapshot_fn,reference,timeout=60,interval=5):
    if reference is None:return {'satisfies_policy':False,'reason':'NO_EMPIRICAL_REFERENCE','pids':set()}
    deadline=time.monotonic()+timeout;consecutive=0;last=None
    while True:
        last=verified_release(snapshot_fn(),reference)
        if last['pids']:return last
        consecutive=consecutive+1 if last['satisfies_policy'] else 0
        if consecutive>=2:return last
        if time.monotonic()>=deadline:
            last['satisfies_policy']=False;last['reason']='RELEASE_OBSERVATION_TIMEOUT';return last
        time.sleep(interval)
