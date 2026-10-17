#!/usr/bin/env python3
"""Stage one startup-only effective-config log; no scheduling/steady-state changes."""
import ast,hashlib,json,socket,subprocess
from pathlib import Path
ADDITION='        logger.info(\n            "Native benchmark startup: scheduler_class=%s async_scheduling=%r cudagraph_mode=%s capture_sizes=%s enforce_eager=%r",\n            type(self).__name__, vllm_config.scheduler_config.async_scheduling,\n            vllm_config.compilation_config.cudagraph_mode,\n            vllm_config.compilation_config.cudagraph_capture_sizes,\n            vllm_config.model_config.enforce_eager,\n        )\n'

def main():
 assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
 target='/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py'
 raw=subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--read-only','--entrypoint','python3','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83','-c','from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(target)+').read_bytes())'],timeout=30)
 assert hashlib.sha256(raw).hexdigest()=='c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884'
 marker='        self.vllm_config = vllm_config\n';text=raw.decode();assert text.count(marker)==1
 text=text.replace(marker,marker+ADDITION);ast.parse(text)
 root=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/native-startup-config-v1');root.mkdir(exist_ok=False)
 path=root/'scheduler.py';path.write_text(text);assert hashlib.sha256(path.read_bytes()).hexdigest()=='f86d9f04af2ecbe9ba08d9ab8e06dd703be9e5c8ad064a6dc2c2aeb771bacf3b'
 result={'source':str(path),'target':target,'sha256':'f86d9f04af2ecbe9ba08d9ab8e06dd703be9e5c8ad064a6dc2c2aeb771bacf3b','original_sha256':'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884','scope':'One info log in Scheduler constructor only, after effective scheduler choice. No step/forward change, payload audit, kernel, precision or RPC. Same file on native eager and graph. No timed-window instrumentation.'}
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
