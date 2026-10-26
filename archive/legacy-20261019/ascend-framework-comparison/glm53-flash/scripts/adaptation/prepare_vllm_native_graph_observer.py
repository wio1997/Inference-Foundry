# SPDX-License-Identifier: Apache-2.0
"""Source-only actual replay observer; no native math/graph changes."""
import argparse,ast,pathlib,hashlib,json,difflib
P=pathlib.Path(__file__).resolve().parents[2];a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a=a.parse_args();a.output.mkdir(parents=True,exist_ok=False)
s=P/'runs/20261017-r124-flash-vllm-performance-preflight/image-sources/vllm_ascend/compilation/acl_graph.py';original=s.read_text();candidate=original.replace('import dataclasses\n','import dataclasses\nimport os\n',1)
anchor='        logger.info_once("ACL graph replay is active (logged once).")\n'
addition='''        if os.environ.get("FLASH_NATIVE_GRAPH_OBSERVER", "0") == "1":
            from vllm_ascend.distributed.kv_transfer.utils.flash_native_graph_observer import graph_replayed

            graph_replayed(self.runtime_mode.name, batch_descriptor, bool(_EXTRA_CTX.is_draft_model))
'''
assert candidate.count(anchor)==1;candidate=candidate.replace(anchor,addition+anchor)
stripped=candidate.replace(addition,'').replace('import dataclasses\nimport os\n','import dataclasses\n',1);assert ast.dump(ast.parse(stripped))==ast.dump(ast.parse(original));ast.parse(candidate)
(a.output/'acl_graph.py').write_text(candidate);(a.output/'acl_graph.py.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='a/acl_graph.py',tofile='b/acl_graph.py')))
helper='''# SPDX-License-Identifier: Apache-2.0
"""Qualification metadata after native target/draft graph replay only.
No tensor access/copy/hash/synchronization. Never enabled for formal timing.
"""
import os,json,pathlib,time,threading
_LOCK=threading.Lock()

def graph_replayed(mode,batch,is_draft):
    if os.environ.get('FLASH_NATIVE_GRAPH_OBSERVER','0')!='1':raise ValueError('Graph qualification observer called while disabled')
    root=pathlib.Path('/run-evidence');phase=None
    p=root/'current-case.json'
    if p.exists():
        v=json.loads(p.read_text());phase={k:v[k] for k in ['case_id','phase','global_dp_rank'] if k in v and type(v[k]) in (str,int)}
    row={'project_id':'glm53-flash','kind':'native_graph_replay_after_branch','pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),'runtime_mode':str(mode),'is_draft':bool(is_draft),'global_dp_rank_env':os.environ.get('FLASH_GLOBAL_DP_RANK'),'request_phase':phase,'scope':'Actual native replay invoked; correlate same-node active requestinterval andselectedDP rank. Startupreplay alone not functional proof.'}
    for name in ['num_tokens','num_reqs','uniform','has_lora']:
        value=getattr(batch,name,None);row[name]=value if type(value) in (bool,int) else None
    target=root/'qualification-metadata';target.mkdir(exist_ok=True);f=target/('graph-'+str(os.getpid())+'.jsonl')
    with _LOCK:
        if f.exists() and f.stat().st_size>8*1024*1024:raise RuntimeError('Qualification metadata bound exceeded')
        with f.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\\n')
'''
ast.parse(helper);(a.output/'flash_native_graph_observer.py').write_text(helper)
manifest={'project_id':'glm53-flash','status':'SOURCE_ONLY_NOT_RUNTIME_QUALIFIED','base_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir())},'math_provenance':'Removeaddedimport/guardedmetadataafteractualreplay only -> ASTidenticalnative. Doesnotchangecapture/replay dispatch/sync/output/target-vs-draft branch.','qualification_env':'FLASH_NATIVE_GRAPH_OBSERVER=1 +FLASH_GLOBAL_DP_RANK, phasefile andsame-node HTTPtimewindow; startupdoesnotcount.','formal_env':'FLASH_NATIVE_GRAPH_OBSERVER=0; nofileIO/tensorCPUaudits/noobservationcalls.'};(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
