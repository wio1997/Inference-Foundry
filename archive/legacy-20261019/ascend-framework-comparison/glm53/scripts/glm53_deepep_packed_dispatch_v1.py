"""D-only lossless raw-byte packing for accepted normal INT8 dispatch."""
import ast,copy,hashlib,inspect,json,os,textwrap
from pathlib import Path
SOURCE_SHA='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
REVISION='glm53-C32-normal-packed-dispatch-v1'
def pack_payload(torch,tokens,scales):
 assert tokens.dtype==torch.int8 and tokens.ndim==2 and tokens.is_contiguous()
 assert scales.dtype==torch.float32 and scales.numel()==tokens.shape[0] and scales.device==tokens.device
 n,h=tokens.shape
 assert h%4==0,'Hidden bytes must preserve FP32 alignment'
 scale_bytes=scales.contiguous().reshape(n).view(torch.int8).reshape(n,4)
 return torch.cat((tokens,scale_bytes),dim=1)
def unpack_payload(torch,packed,hidden):
 assert packed.dtype==torch.int8 and packed.ndim==2 and packed.shape[1]==hidden+4 and hidden%4==0
 n=packed.shape[0]
 # No float arithmetic; both outputs retain ownership even if contiguous is an alias.
 tokens=packed[:,:hidden].contiguous()
 scales=packed[:,hidden:].contiguous().view(torch.float32).reshape(n)
 return tokens,scales
def _record(self,tokens,scales,packed):
 calls=getattr(self,'_glm53_packed_dispatch_calls',0)+1;self._glm53_packed_dispatch_calls=calls
 if calls==1:
  print('GLM53_C32_PACKED_DISPATCH_FIRST_USE='+json.dumps({'project_id':'glm53','revision':REVISION,'pid':os.getpid(),'node':os.environ.get('GLM53_NODE'),'device_index':tokens.device.index,'group_size':self.group_size,'rows':tokens.shape[0],'hidden':tokens.shape[1],'token_dtype':str(tokens.dtype),'scale_dtype':str(scales.dtype),'wire_dtype':str(packed.dtype),'wire_columns':packed.shape[1],'payload_collectives_before':2,'payload_collectives_after':1,'scale_bytes_per_token':4,'scale_arithmetic':False,'upstream_normal_strategy_sha256':SOURCE_SHA}),flush=True)
def install(module):
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA,'Unsupported native source'
 cls=module.AlltoAllNormalCommStrategy
 assert getattr(cls,'_glm53_metadata_batch_installed',False),'Install accepted L04 first'
 assert not getattr(cls,'_glm53_packed_dispatch_installed',False),'Do not install twice'
 original_dispatch=cls._glm53_original_metadata_methods[1]
 tree=ast.parse(textwrap.dedent(inspect.getsource(original_dispatch)));assert len(tree.body)==1
 fn=tree.body[0];assert isinstance(fn,ast.FunctionDef) and fn.name=='dispatch'
 # Guard exact two-collective original transport block before any replacement.
 first=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.If) and ast.unparse(x.test)=='use_quant == 1')
 oldblock=fn.body[first:first+4]
 expected=ast.parse("""
if use_quant == 1:
 _, dynamic_scale_after_all2all, scale_handle = self._async_all_to_all(dynamic_scale, output_splits, input_splits, self.group)
 scale_handle.wait()
 dynamic_scale.untyped_storage().resize_(0)
_, global_input_tokens, handle_a2a = self._async_all_to_all(permutated_tokens, output_splits, input_splits, self.group)
handle_a2a.wait()
permutated_tokens.untyped_storage().resize_(0)
""").body
 assert len(oldblock)==4 and [ast.dump(x) for x in oldblock]==[ast.dump(x) for x in expected],'Native transport changed'
 replacement=ast.parse("""
if use_quant == 1:
 _glm53_packed_input = _glm53_pack_payload(torch,permutated_tokens,dynamic_scale)
 _, _glm53_packed_output, _glm53_packed_handle = self._async_all_to_all(_glm53_packed_input,output_splits,input_splits,self.group)
 _glm53_packed_handle.wait()
 global_input_tokens,dynamic_scale_after_all2all = _glm53_unpack_payload(torch,_glm53_packed_output,permutated_tokens.shape[1])
 _glm53_record_packed_dispatch(self,permutated_tokens,dynamic_scale,_glm53_packed_input)
 dynamic_scale.untyped_storage().resize_(0)
 permutated_tokens.untyped_storage().resize_(0)
 del _glm53_packed_input,_glm53_packed_output
else:
 _, global_input_tokens, handle_a2a = self._async_all_to_all(permutated_tokens,output_splits,input_splits,self.group)
 handle_a2a.wait()
 permutated_tokens.untyped_storage().resize_(0)
""").body
 fn.body[first:first+4]=replacement
 # Same accepted L04 cached host-count substitution; no new metadata download.
 target=ast.parse('num_global_tokens_per_local_expert.sum(axis=0).cpu().numpy().tolist()',mode='eval').body
 matches=[]
 class Cached(ast.NodeTransformer):
  def visit_Call(self,node):
   if ast.dump(node)==ast.dump(target):
    matches.append(node);return ast.parse("list(self._alltoall_layout['_glm53_host_expert_counts'])",mode='eval').body
   return self.generic_visit(node)
 fn=Cached().visit(fn);assert len(matches)==1
 compiled=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={}
 module.__dict__.update(_glm53_pack_payload=pack_payload,_glm53_unpack_payload=unpack_payload,_glm53_record_packed_dispatch=_record)
 exec(compile(compiled,str(module.__file__)+':'+REVISION,'exec'),module.__dict__,scope)
 cls._glm53_unpacked_dispatch=cls.dispatch;cls.dispatch=scope['dispatch'];cls._glm53_packed_dispatch_installed=True
 return {'revision':REVISION,'upstream_sha256':SOURCE_SHA,'method':'dispatch','changed_transport':'int8 token+raw fp32 scale bytes;2collectives->1','unchanged':['L04layout','BF16transport','combine','quantization/routing/GMM arithmetic']}
def uninstall(module):
 cls=module.AlltoAllNormalCommStrategy;assert cls._glm53_packed_dispatch_installed
 cls.dispatch=cls._glm53_unpacked_dispatch;cls._glm53_packed_dispatch_installed=False
