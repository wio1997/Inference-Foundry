#!/usr/bin/env python3
import json,sys
from types import SimpleNamespace as NS
from L034_extension_components_v4 import R,ns,registry,load,import_source,records
O=R/'runs/L034-R009-lora-gates';O.mkdir(parents=True,exist_ok=False);m=import_source('sglang.srt.lora.lora_config','python/sglang/srt/lora/lora_config.py');load('python/sglang/srt/lora/lora_manager.py',methods={'LoRAManager':['validate_new_adapter']});Manager=ns['LoRAManager'];checks=[]
def test(name,cfg,pinned=False,n=0,slots=3,existing=(),want=True):
 manager=Manager();manager.lora_refs={str(i):r for i,r in enumerate(existing)};manager.num_pinned_loras=n;manager.max_loras_per_batch=slots;ref=registry.LoRARef(lora_name='new',lora_path='/synthetic/new',pinned=pinned);err=None
 try:manager.validate_new_adapter(cfg,ref)
 except ValueError as e:err=str(e)
 checks.append({'name':name,'pass':(err is None)==want,'metadata_allowed':err is None,'reason':err,'memory_pool_instantiated':False})
def config(**kw):
 values=dict(target_modules=['q_proj'],r=4,lora_alpha=8);values.update(kw);return m.LoRAConfig.from_dict(values)
test('metadata clean',config());test('DoRA denied',config(use_dora=True),want=False)
a=m.LoRAConfig.from_dict(dict(target_modules=['q_proj'],r=4,lora_alpha=8),added_tokens_config={'old':99},base_vocab_size=100);test('existing vocab token filtered',a)
a=m.LoRAConfig.from_dict(dict(target_modules=['q_proj'],r=4,lora_alpha=8),added_tokens_config={'new':100},base_vocab_size=100);test('added vocab token denied',a,want=False)
test('pin preserves final slot',config(),pinned=True,n=2,slots=3,want=False);test('pin when room remains',config(),pinned=True,n=1,slots=3);test('unpinned accepted with pinned atlimit',config(),pinned=False,n=2,slots=3);test('single slot cannotpin',config(),pinned=True,n=0,slots=1,want=False);test('duplicate name denied',config(),existing=[registry.LoRARef(lora_name='new',lora_path='/synthetic/old')],want=False)
failed=[c for c in checks if not c['pass']];j={'status':'FAIL' if failed else 'PASS','semantic_checks':len(checks),'checks':checks,'source_components':records,'scope':'Real LoRAConfig.from_dict and original validate_new_adapter metadata gate only. Manager constructor/memory_pool absent; no shape/rank pool support, weight/kernel load, NPU/PD/MTP or throughput proven.','native_executed':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':j['status'],'semantic_checks':len(checks)}));sys.exit(1 if failed else 0)
