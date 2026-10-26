#!/usr/bin/env python3
import hashlib,json,pathlib,sys,traceback
from array import array
from types import SimpleNamespace as NS
from unittest.mock import patch
from L037_candidate_components import R,storage,ns,records,config,controller,attachment,BOUNDARIES,torch
O=R/'runs/L037-R004-storage-candidate';O.mkdir(parents=True,exist_ok=False);scratch=O/'owned-files';scratch.mkdir();checks=[];cases=[];stage='setup';error=None
def ck(name,value,**detail):checks.append({'stage':stage,'name':name,'pass':bool(value),'detail':detail})
def read(store,key,like):return store.get(key,torch.empty_like(like))
def owned(root):
 root=root.resolve();assert root.is_relative_to(scratch.resolve());root.mkdir(exist_ok=False);return root
try:
 manifest=json.loads((R/'fixtures/M001-v1/manifest.json').read_text());fixtures={x['label']:x for x in manifest['fixtures']}
 for label in ['c2-short-0','c3-long-0']:
  f=fixtures[label];p=R/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];ids=json.loads(p.read_text())['input_ids'];assert len(ids)==f['input_tokens'];key=hashlib.sha256(array('q',ids).tobytes()).hexdigest();data=torch.tensor(ids[:128],dtype=torch.int64);new=data+1
  for mla in [False,True]:
   for metadata in [False,True]:
    for kind in ['model-name-alias','distinct-name','clear-failure']:
     stage=str((label,mla,metadata,kind));root=owned(scratch/str(len(cases)));name_a='org/A';name_b='org-A' if kind=='model-name-alias' else 'org/B';a=storage.HiCacheFile(config(name_a,mla,metadata),str(root));b=storage.HiCacheFile(config(name_b,mla,metadata),str(root));ck('whole File and LRU unbounded config',not a._evictor.configured and a.enable_metadata_cache==metadata);ck('actual CPU file write',a.set(key,data));target=read(a,key,data);ck('same config bytes roundtrip',target is not None and torch.equal(target,data))
     if kind in ('model-name-alias','distinct-name'):
      hit=read(b,key,data);same=a._get_suffixed_key(key)==b._get_suffixed_key(key);ck('namespace path matches read observation',(hit is not None)==same);ck('CONTRACT distinct configured model names must not share stored bytes',hit is None,model_a=name_a,model_b=name_b,same_file=same,loaded_old_bytes=hit is not None and torch.equal(hit,data));ck('immutable same-key write returns success',a.set(key,new));persisted=read(a,key,data);ck('existing immutable key preserves old bytes',persisted is not None and torch.equal(persisted,data));fresh=storage.HiCacheFile(config(name_a,mla,metadata),str(root));again=read(fresh,key,data);ck('new backend instance reads retained bytes',again is not None and torch.equal(again,data));other=storage.HiCacheFile(config(name_a,mla,metadata,tp_rank=1),str(root));rankhit=read(other,key,data);ck('rank suffix follows MLA sharing contract',(rankhit is not None)==mla)
     else:
      ck('other namespace writes independent file',b.set(key,new));victim=(root/(a._get_suffixed_key(key)+'.bin')).resolve();assert victim.is_file();real_remove=storage.os.remove;attempts=[]
      def fail_owned(path):
       p=pathlib.Path(path).resolve();assert p.is_relative_to(root);attempts.append(str(p.name))
       if p==victim:raise OSError('injected delete failure for one task-owned file')
       return real_remove(path)
      with patch.object(storage.os,'remove',side_effect=fail_owned):
       raw=a.clear();reported=attachment(controller(a)).clear()
      ck('actual File clear reports failure',raw is False and victim.exists());ck('CONTRACT attachment must report backend clear failure',reported is False,backend_clear=raw,reported_success=reported,retained_file=victim.name);remaining=read(a,key,data);ck('failed clear retained actual old bytes',remaining is not None and torch.equal(remaining,data));ck('owned fault injected on repeated call',attempts.count(victim.name)==2)
     ck('normal clear succeeds',attachment(controller(a)).clear() is True);ck('clear removes all files in configured owned root',not list(root.iterdir()));ck('other namespace read misses after root clear',read(b,key,data) is None);cases.append({'kind':kind,'input_fixture':label,'input_tokens':len(ids),'payload_elements':len(data),'payload_dtype':str(data.dtype),'payload_sha256':hashlib.sha256(data.numpy().tobytes()).hexdigest(),'key_fixture_sha256':key,'MLA_config':mla,'metadata_cache':metadata,'model_a':name_a,'model_b':name_b,'same_suffix':a.config_suffix==b.config_suffix,'scratch_root':str(root.relative_to(R)),'remaining_files':len(list(root.iterdir())),'native':False})
 stage='clear-return-contracts'
 for value in [None,True,False]:
  backend=NS(clear=lambda value=value:value);reported=attachment(controller(backend)).clear();ck('CONTRACT explicit False fails while None/True compatible '+str(value),reported==(value is not False),backend_return=value,reported=reported)
 def raised():raise OSError('explicit backend exception fixture')
 ck('raised backend failure reported',attachment(controller(NS(clear=raised))).clear() is False);ck('disabled backend rejected',attachment(controller(NS(clear=lambda:None),False)).clear() is False);ck('missing clear rejected',attachment(controller(NS())).clear() is False);cases.append({'kind':stage,'native':False})
 stage='same-backend-attach'
 c=controller(NS());calls=[];c.attach_storage_backend=lambda **kw:calls.append(kw);c.storage_config=config('old-model');cache=NS(cache_controller=c,enable_storage=True);att=ns['StorageAttachment'](cache);ok,msg=att.attach('file',served_model_name='new-model',storage_backend_extra_config_json='{"extra_backend_tag":"new-generation"}',hicache_write_policy='write_back');ck('same backend attach only updates policies',ok and not calls and c.storage_config.model_name=='old-model' and c.write_policy=='write_back');ck('different backend requires detach',att.attach('mooncake')[0] is False and not calls);before=c.write_policy;ck('invalid policy rejected without change',att.attach('file',hicache_write_policy='invalid')[0] is False and c.write_policy==before);cases.append({'kind':stage,'native':False})
 stage='NPU-key-helper'
 keys=['fixture-page-hash'];a=ns['NpuMemcacheStore']();a.storage_config=config('model-A');a.extra_backend_tag=None;a._init_runtime_fields(a.storage_config);b=ns['NpuMemcacheStore']();b.storage_config=config('model-B');b.extra_backend_tag=None;b._init_runtime_fields(b.storage_config);ck('model name alone does not change NPU tag helper',a._tag_keys(keys)==b._tag_keys(keys) and a.mha_suffix==b.mha_suffix);a.extra_backend_tag='generation-A';b.extra_backend_tag='generation-B';ck('explicit tag changes NPU helper keys',a._tag_keys(keys)!=b._tag_keys(keys));cases.append({'kind':stage,'constructor_executed':False,'native':False})
 stage='weight-update-dispatch';w=ns['SchedulerWeightUpdaterManager']();calls=[];w.flush_cache=lambda **kw:calls.append(kw) or True;w.flush_cache_after_weight_update(NS(flush_cache=False,torch_empty_cache=False));ck('optional flush skipped',not calls);w.flush_cache_after_weight_update(NS(flush_cache=True,torch_empty_cache=False));ck('flush callback gets requested flag',calls==[{'empty_cache':False}]);w.flush_cache=lambda **kw:False
 try:w.flush_cache_after_weight_update(NS(flush_cache=True,torch_empty_cache=True))
 except AssertionError:ck('failed memory flush raises',True)
 else:ck('failed memory flush raises',False)
 versions=[];w.scheduler=NS(record_weight_version_change=lambda **kw:versions.append(kw));w.record_weight_version_after_update('revision-B');ck('version callback dispatch',versions==[{'new_version':'revision-B'}]);cases.append({'kind':stage,'weights_updated':False,'native':False})

except Exception as e:error={'stage':stage,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
failed=[x for x in checks if not x['pass']];result={'status':'INVALID' if error else 'FAIL' if failed else 'PASS','harness_error':error,'case_count':len(cases),'semantic_checks':len(checks),'failed_checks':len(failed),'checks':checks,'cases':cases,'source_components':records,'boundaries':BOUNDARIES,'native_executed':False,'current_deployment_fault_proven':False,'candidate_accepted':False,'production_applied':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','case_count','semantic_checks','failed_checks','harness_error']}));sys.exit(2 if error else 1 if failed else 0)
