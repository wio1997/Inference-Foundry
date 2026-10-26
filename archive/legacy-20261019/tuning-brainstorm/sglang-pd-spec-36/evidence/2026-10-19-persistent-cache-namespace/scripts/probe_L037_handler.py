#!/usr/bin/env python3
import json,pathlib,sys,traceback
from types import SimpleNamespace as NS
from unittest.mock import patch
from L037_clear_handler_components import R,ns,records,config,storage,controller,attachment,torch,msgspec,BOUNDARIES
O=R/'runs/L037-R005-clear-handler';O.mkdir(parents=True,exist_ok=False);scratch=O/'owned-files';scratch.mkdir();checks=[];cases=[];stage='setup';error=None
def ck(name,value,**detail):checks.append({'stage':stage,'name':name,'pass':bool(value),'detail':detail})
def call(c,enabled=True):
 trace=[];a=attachment(c)
 def clear():value=a.clear();trace.append(value);return value
 s=ns['Scheduler']();s.enable_hierarchical_cache=enabled;s.tree_cache=NS(clear_storage_backend=clear);out=s.clear_hicache_storage_wrapped(ns['ClearHiCacheReqInput'](rid='cpu-clear'));wire=msgspec.msgpack.encode(out);decoded=msgspec.msgpack.decode(wire,type=ns['ClearHiCacheReqOutput']);ck('real output msgpack preserves response',decoded.success==out.success);return out,trace
try:
 for metadata in [False,True]:
  for fault in [False,True]:
   stage=str(('owned-file',metadata,fault));root=(scratch/str(len(cases))).resolve();assert root.is_relative_to(scratch.resolve());a=storage.HiCacheFile(config('model-A',metadata=metadata),str(root));key='controlled-page-key';data=torch.arange(128,dtype=torch.int64);ck('actual owned file write',a.set(key,data));victim=root/(a._get_suffixed_key(key)+'.bin');real_remove=storage.os.remove
   def injected(path):
    p=pathlib.Path(path).resolve();assert p.is_relative_to(root)
    if fault and p==victim:raise OSError('owned delete failure')
    return real_remove(path)
   with patch.object(storage.os,'remove',side_effect=injected):out,trace=call(controller(a))
   ck('controller-only patch delivers real File result',trace==[not fault]);ck('CONTRACT scheduler response preserves clear outcome',out.success==(not fault),lower_ack=trace[0],response=out.success,retained=victim.exists());ck('disk agrees with delete fault',victim.exists()==fault)
   if fault:target=a.get(key,torch.empty_like(data));ck('failure retains actual old payload',target is not None and torch.equal(target,data))
   ck('normal cleanup removes owned files',a.clear() and not list(root.iterdir()));cases.append({'kind':'actual-file','metadata':metadata,'fault_injected':fault,'lower_ack':trace[0],'response':out.success,'remaining_files':len(list(root.iterdir())),'native':False})
 stage='storage-disabled';out,trace=call(controller(NS(clear=lambda:None),False));ck('CONTRACT disabled storage response is failure',out.success is False and trace==[False],response=out.success);cases.append({'kind':stage,'native':False})
 stage='feature-disabled';out,trace=call(controller(NS(clear=lambda:None)),False);ck('disabled hierarchical feature skips lower call',out.success is False and not trace);cases.append({'kind':stage,'native':False})
 for value in [None,True,False]:
  stage='backend-return-'+str(value);out,trace=call(controller(NS(clear=lambda value=value:value)));ck('CONTRACT scheduler preserves None/True/False compatibility',out.success==(value is not False),backend_return=value,lower_ack=trace[0],response=out.success);cases.append({'kind':stage,'native':False})
 stage='backend-exception'
 def raised():raise OSError('backend exception fixture')
 out,trace=call(controller(NS(clear=raised)));ck('CONTRACT scheduler reports caught backend exception',out.success is False and trace==[False],response=out.success);cases.append({'kind':stage,'native':False})
except Exception as e:error={'stage':stage,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
failed=[x for x in checks if not x['pass']];result={'status':'INVALID' if error else 'FAIL' if failed else 'PASS','harness_error':error,'case_count':len(cases),'semantic_checks':len(checks),'failed_checks':len(failed),'checks':checks,'cases':cases,'source_components':records,'boundaries':BOUNDARIES,'native_executed':False,'current_deployment_fault_proven':False,'candidate_accepted':False,'production_applied':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','case_count','semantic_checks','failed_checks','harness_error']}));sys.exit(2 if error else 1 if failed else 0)
