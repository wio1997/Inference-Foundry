"""Export the one completed capture after original health is restored. CPU only."""
from pathlib import Path
import json,subprocess,sys,time,urllib.request

root=Path(sys.argv[1]);state=root/'export-state.json'
def write(status,**extra):state.write_text(json.dumps({'status':status,**extra},indent=2))
def healthy():
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:30075/health',timeout=5) as r:return r.status==200
    except Exception:return False
try:
    write('WAITING_FOR_CAPTURE_AND_RESTORE')
    for _ in range(720):
        if not (root/'receipt.json').exists():
            time.sleep(1)
            continue
        receipt=json.load(open(root/'receipt.json'))
        if receipt.get('status')=='FAILED_RESTORED':raise RuntimeError('capture failed; restored; inspect receipt')
        if receipt.get('status')=='CAPTURED_RESTORED' and healthy():break
        time.sleep(5)
    else:raise RuntimeError('capture/restore deadline exceeded')
    croot='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/'+root.name
    (root/'parser-logs').mkdir(exist_ok=True)
    write('EXPORTING_EXISTING_TRACES')
    with (root/'offline-export.log').open('wb') as log:
        r=subprocess.run(['docker','exec','-e','ASCEND_PROCESS_LOG_PATH='+croot+'/parser-logs','-e','ASCEND_WORK_PATH='+croot+'/parser-logs','-e','PYTHONDONTWRITEBYTECODE=1','wio-vllm-v41-repro171-20261102','python3','-c','from torch_npu.profiler import profiler; profiler.analyse('+repr(croot+'/traces')+', max_process_number=4)'],stdout=log,stderr=subprocess.STDOUT,timeout=1800);r.check_returncode()
    write('ANALYZING_SCOPES')
    with (root/'scope-analysis.log').open('wb') as log:
        r=subprocess.run(['python3','-B','-I','-S',str(root/'analyze_scopes.py'),str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=1200);r.check_returncode()
    write('ANALYZING_FUNCTIONS')
    with (root/'function-analysis.log').open('wb') as log:
        r=subprocess.run(['python3','-B','-I','-S',str(root/'analyze_functions.py'),str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=1200);r.check_returncode()
    write('ANALYZING_DEVICE_DETAIL')
    with (root/'device-analysis.log').open('wb') as log:
        r=subprocess.run(['python3','-B','-I','-S',str(root/'analyze_detailed.py'),str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=1200);r.check_returncode()
    write('EXPORTED_AND_ANALYZED',worker_traces=16)
except BaseException as e:
    write('ERROR',error=str(e));raise
