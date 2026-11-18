"""Run fixed-source control branches on CPU; no imports of vLLM or torch."""
from pathlib import Path
import ast,copy,hashlib,json,types
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parent/'G359_fixed_vllm_connectors');parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'G365_vllm_connector_contract.json');args=parser.parse_args();B=args.source_root.resolve(strict=True);hashes={}
def fn(path,name,cls=None,namespace=None):
 p=B/path;hashes[path]=hashlib.sha256(p.read_bytes()).hexdigest();tree=ast.parse(p.read_text());body=tree.body
 if cls:body=next(n.body for n in body if isinstance(n,ast.ClassDef) and n.name==cls)
 node=copy.deepcopy(next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name));node.decorator_list=[]
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]);ns=dict(namespace or {});exec(compile(ast.fix_missing_locations(mod),'fixed_native_contract','exec'),ns);return ns[name]
N=types.SimpleNamespace;cases=[];keys=object()
storepath='vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py'
configure=fn(storepath,'configure_dspark_prefix_cache','AscendStoreConnector',{'DSparkPrefixKeys':N(from_config=lambda cfg:keys)})
for layerwise,plane,role,want in [(False,'gva','kv_producer',False),(True,'gva','kv_producer',True),(True,'block_key','kv_producer',False),(True,'gva','kv_consumer',False)]:
 calls=[];scheduler=N(use_eagle=False);worker=N(configure_dspark_prefix_cache=lambda k:calls.append(k));self=N(use_layerwise=layerwise,layerwise_data_plane=plane,kv_role=role,connector_scheduler=scheduler,connector_worker=worker,dspark_prefix_cache_enabled=False)
 result=configure(self,object());assert result is want;assert not want or scheduler.use_eagle and calls==[keys];cases.append({'case':'store_configure','use_layerwise':layerwise,'data_plane':plane,'role':role,'enabled':result})
piecewise=fn(storepath,'requires_piecewise_for_cudagraph','AscendStoreConnector')
assert piecewise(None,{'use_layerwise':True}) and not piecewise(None,{'use_layerwise':False});cases.append({'case':'layerwise_graph_policy','layerwise_requires_piecewise':True,'nonlayerwise_requires_piecewise':False})
p='vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py';uses=fn(p,'uses_sfa_dspark_kv_transfer');get_aux=fn(p,'get_pd_dspark_aux_layer_ids',namespace={'uses_sfa_dspark_kv_transfer':uses,'get_dspark_aux_layer_ids':lambda cfg:(17,)})
def config(pcp=1,prefix=False,eager=True,connector='SfaRemoteD2HConnector'):
 child={'kv_connector':connector,'kv_connector_extra_config':{}};return N(speculative_config=N(method='dspark'),kv_transfer_config=N(is_kv_producer=True,is_kv_consumer=False,kv_connector='MultiConnector',kv_connector_extra_config={'connectors':[child]}),parallel_config=N(prefill_context_parallel_size=pcp,decode_context_parallel_size=1),model_config=N(enforce_eager=eager),cache_config=N(enable_prefix_caching=prefix))
for kw,text in [({'pcp':4},'does not support context parallelism'),({'prefix':True},'local prefix caching disabled'),({'eager':False},'requires eager prefill')]:
 try:get_aux(config(**kw));raise RuntimeError('Expected native rejection')
 except ValueError as e:assert text in str(e);cases.append({'case':'sfa_P_guard','config':kw,'native_error':str(e)})
assert get_aux(config())==(17,);assert not uses(config(connector='MooncakeHybridConnector'));assert get_aux(config(connector='MooncakeHybridConnector'))==();cases.append({'case':'native_valid_P_control','CP1_eager_localprefix_off_accepted':True,'generic_Mooncake_reference_does_not_select_SFA_draft_protocol':True})
r={'kind':'CPU_fixed_native_branch_contract','source_SHA256':hashes,'cases':cases,'passed':True,'fixture_scope':'DSparkPrefixKeys.from_config and auxiliary IDs use recording interface fixtures; no real key hashes, tensors, SDK/store, PD, graph, loaded weights or performance acceptance','conclusion':'Reference MultiConnector mechanism exists; reference nonlayerwiseMooncake doesnot optinto this fixed pairedDSpark prefix path. Preserve native P localprefix/CP/eager guards; P GVA store layerwise and D fullgraphs require role-specific graph policy.'};args.output.write_text(json.dumps(r,indent=2));print(json.dumps({'passed':True,'cases':len(cases),'source_functions':len(hashes),'scope':r['fixture_scope']}))
