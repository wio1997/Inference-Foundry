"""Job-scoped CANN CLI launch adapter; all data processing stays in pinned CANN.

The native frontend stalls without compute-device nodes. The SDK still sees its
real installation path. Only its known metadata/export/analyze invocations are
translated to the same installation's Python backend. No profiler data changes.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

if os.environ.get('DS41_CPU_CANN_ADAPTER') == 'g853':
    _original_popen = subprocess.Popen
    _manifest_path = Path(__file__).with_name('native_CANN_help_g850.json')
    _manifest = json.loads(_manifest_path.read_text())
    _binary = Path(_manifest['msprof']).resolve()
    _backend = Path(_manifest['python_cli'])
    assert _manifest['returncode'] == 0 and '--type' in _manifest['stdout']

    def _translate(args):
        if not isinstance(args,(list,tuple)) or not args:
            return args
        if str(args[0]) != _manifest['msprof'] and str(args[0]) != str(_binary):
            return args
        assert hashlib.sha256(_binary.read_bytes()).hexdigest()==_manifest['msprof_sha256']
        assert hashlib.sha256(_backend.read_bytes()).hexdigest()==_manifest['python_cli_sha256']
        flags=list(map(str,args[1:]))
        if flags==['--help']:
            replacement=[sys.executable,'-c','import json,sys;sys.stdout.write(json.load(open(sys.argv[1]))["stdout"])',str(_manifest_path)]
            kind='cached_actual_native_help_same_image_and_SHA'
        else:
            assert len(flags)==3 and flags[1]=='--type=db' and flags[2].startswith('--output='),('Unsupported msprof invocation',flags)
            output=flags[2].split('=',1)[1]
            if flags[0]=='--export=on':
                replacement=[sys.executable,str(_backend),'export','db','-dir='+output]
            elif flags[0]=='--analyze=on':
                replacement=[sys.executable,str(_backend),'analyze','-dir='+output,'--rule=communication,communication_matrix','--type=db']
            else:
                raise AssertionError(('Unsupported msprof action',flags))
            kind='same_installed_CANN_Python_backend_full_db'
        log=Path(os.environ['DS41_CPU_CLI_LOG_DIR'])/('commands_'+str(os.getpid())+'.jsonl')
        with log.open('a') as stream:
            stream.write(json.dumps({'time_ns':time.time_ns(),'pid':os.getpid(),'kind':kind,'original':list(args),'translated':replacement})+'\n')
        return replacement

    class CANNBackendPopen(_original_popen):
        def __init__(self,args,*positional,**kwargs):
            super().__init__(_translate(args),*positional,**kwargs)

    subprocess.Popen=CANNBackendPopen
