from pathlib import Path
from contextlib import AbstractContextManager,nullcontext
from types import SimpleNamespace as N
from typing import cast
import ast,copy,hashlib,json
T=Path(__file__).resolve().parent
paths={'fixed0.30':T/'fixed/patch_dspark.py','PR12':T/'reference/patch_dspark.py'};results=[];pins={}
for label,p in paths.items():
 pins[label]=hashlib.sha256(p.read_bytes()).hexdigest();tree=ast.parse(p.read_text());f=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_load_dspark_model_with_target_quant'));f.decorator_list=[]
 for same in [True,False]:
  for fail in [False,True]:
   target_quant=object();draft_quant=object();old_get=lambda cfg:draft_quant;old_share=lambda *a:None;mu=N(get_draft_quant_config=old_get);eu=N(maybe_share_target_embed=old_share);seen=[]
   def loader(target,cfg):
    seen.append(mu.get_draft_quant_config(cfg))
    if fail:raise RuntimeError('intentional_loader_failure')
    return 'loaded_fixture'
   ns=dict(model_utils=mu,eagle_utils=eu,_original_get_draft_quant_config=old_get,_original_maybe_share_target_embed=old_share,_pp_safe_maybe_share_target_embed=lambda *a:None,_original_load_dspark_model=loader,resolve_spec_pp_support=lambda cfg:None,use_legacy_spec_pp=lambda:False,bypass_upstream_spec_pp_guard=lambda *a:nullcontext(),cast=cast,AbstractContextManager=AbstractContextManager,nullcontext=nullcontext)
   unit=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]));exec(compile(unit,'actual_native_quant_contract','exec'),ns)
   cfg=N(quant_config=target_quant,model_config=N(model='same-checkpoint'),speculative_config=N(draft_model_config=N(model='same-checkpoint' if same else 'separate-draft',hf_config=N())))
   try:assert ns[f.name](object(),cfg)=='loaded_fixture' and not fail
   except RuntimeError as e:assert fail and str(e)=='intentional_loader_failure'
   assert seen==[target_quant if same else draft_quant];assert mu.get_draft_quant_config is old_get and eu.maybe_share_target_embed is old_share
   results.append({'source':label,'same_checkpoint':same,'inner_loader_failure':fail,'correct_quant_object_selected':True,'original_hooks_restored':True})
r={'passed':True,'source_sha256':pins,'cases':results,'DeepSeek_PP1_contract_same':True,'scope':'Executes actualnativewrapper bodies on CPU interfacefixtures. No actualModelSlim descriptor parse, checkpointload, PP>1, NPU, graph, PD, quality orperformance acceptance. PR12Minimax-specificdifferences notappliedtoDeepSeek.'};(T/'result.local.json').write_text(json.dumps(r,indent=2));print('Actualfixed/referenceDSparkPP1 quantinheritance contracts passed',len(results),'CPUcases; no newadapter neededfor thismechanism')
