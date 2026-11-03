"""Before any model import: require approval and bound individual output files."""
import json,os,resource,sys
sys.path.insert(0,'/wio-diagnostic')
from shared_runtime import verify_approval
plan=verify_approval()
resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
env=dict(os.environ)
env.update(WIO_BUILTIN_COHORT_ENABLE='1',WIO_TARGETED_DISPATCH_ENABLE='0',WIO_HOST_TRACE_ENABLE='0')
os.execve('/bin/bash',['bash','/wio-diagnostic/serve-diagnostic.sh'],env)
