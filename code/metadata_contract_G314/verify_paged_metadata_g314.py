import ast,hashlib,json,types,copy
from pathlib import Path
from paged_indexer_scope_g314 import permitted,active_v41_npu_indexer
src=Path('../glm53_pd_align_910b_20261116/G156_SG_runtime_transfer/runtime_sglang_g32/sglang/srt/layers/attention/dsv4/metadata.py');digest=hashlib.sha256(src.read_bytes()).hexdigest();assert digest==json.loads(Path('sg_pd_g314/native_backend_sha256.json').read_text())['metadata']
cls=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PagedIndexerMetadata');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__')
class Flag:
 def get(self):return True
ns={'is_hip':lambda:False,'is_xpu':lambda:False,'envs':types.SimpleNamespace(SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=Flag()),'torch':types.SimpleNamespace(empty=lambda shape:('empty',shape)),'_v41_npu_page_guard_g314':permitted}
def compile_fn(f):
 d=dict(ns);exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'native_metadata_fixture','exec'),d);return d['__post_init__']
original=compile_fn(copy.deepcopy(fn));patched=copy.deepcopy(fn);n=0
for a in ast.walk(patched):
 if isinstance(a,ast.Assert) and ast.unparse(a.test)=='self.page_size == 256':a.test=ast.parse('_v41_npu_page_guard_g314(self, envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get())',mode='eval').body;n+=1
assert n==1;updated=compile_fn(patched)
def item(**kw):
 d=dict(page_size=128,compress_ratio=1,compressed_page_size=64,page_table=types.SimpleNamespace(device=types.SimpleNamespace(type='npu')),compressed_seq_lens=types.SimpleNamespace(device=types.SimpleNamespace(type='npu')),force_deep_gemm_metadata=False,use_topk_v2=False);d.update(kw);return types.SimpleNamespace(**d)
try:original(item());raise RuntimeError('Expected original failure')
except AssertionError:pass
cases=[]
for ratio in [1,2]:
 for rows in [1,6]:
  token=active_v41_npu_indexer.set(True)
  try:a=item(compress_ratio=ratio,rows_per_request=rows);updated(a);assert a.deep_gemm_metadata is None;cases.append({'ratio':ratio,'rows_per_request':rows,'passed':True})
  finally:active_v41_npu_indexer.reset(token)
for changes in [{},{'compress_ratio':4},{'compressed_page_size':128},{'page_size':64},{'page_table':types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))}]:
 token=active_v41_npu_indexer.set(bool(changes))
 try:
  try:updated(item(**changes));raise RuntimeError('Guard erroneously accepted '+str(changes))
  except AssertionError:pass
 finally:active_v41_npu_indexer.reset(token)
token=active_v41_npu_indexer.set(True)
try:
 for changes in [{'use_topk_v2':True},{'force_deep_gemm_metadata':True}]:assert not permitted(item(**changes),True)
 assert not permitted(item(),False)
finally:active_v41_npu_indexer.reset(token)
updated(item(page_size=256,compressed_page_size=64,compress_ratio=4))
r={'native_metadata_sha256':digest,'original128_assert_reproduced':True,'native_method_CPU_interface_cases':cases,'out_of_scope_rejected':True,'native256_retained':True,'first_fixture_attempt':'Missing native topk-v2 import in CPU fixture; narrowed to direct negative guard test, retained in task log','real_tensor_NPU_graph_fullmodel_PD_or_performance_accepted':False};Path('G314_paged_metadata_contract.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
