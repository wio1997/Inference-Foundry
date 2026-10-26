"""Task-owned D-only method overlay; original DeepEP package files remain untouched."""
import ast,copy,hashlib,inspect,json,os,textwrap
from pathlib import Path

SOURCE_SHA='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
REVISION='glm53-routing-host-metadata-batch-v1'

def _function(method):
 tree=ast.parse(textwrap.dedent(inspect.getsource(method)));assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
 return tree.body[0]
def _download_input(expression):
 node=expression
 for name in ['tolist','numpy','cpu']:
  assert isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr==name and not node.args and not node.keywords
  node=node.func.value
 return copy.deepcopy(node)
def _assigns(statement,name):
 return isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id==name
def _is_layout(statement):
 return isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Attribute) and statement.targets[0].attr=='_alltoall_layout' and isinstance(statement.targets[0].value,ast.Name) and statement.targets[0].value.id=='self'
def _record(self,elements,device,router_rows):
 calls=getattr(self,'_glm53_metadata_batch_calls',0)+1;self._glm53_metadata_batch_calls=calls
 if calls==1:
  print('GLM53_DEEPEP_METADATA_BATCH_FIRST_USE='+json.dumps({'project_id':'glm53','revision':REVISION,'pid':os.getpid(),'device_index':device,'group_size':self.group_size,'router_rows':router_rows,'host_elements':elements,'host_vectors':3,'metadata_host_downloads':1,'repeat_output_size_from_same_snapshot':True,'upstream_normal_strategy_sha256':SOURCE_SHA}),flush=True)

def install(module):
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA,'Unsupported upstream DeepEP source; refuse overlay'
 cls=module.AlltoAllNormalCommStrategy
 assert not getattr(cls,'_glm53_metadata_batch_installed',False),'Do not install twice'
 original_layout=cls.get_dispatch_layout;original_dispatch=cls.dispatch
 layout=_function(original_layout);dispatch=_function(original_dispatch)
 inputs=[x for x in layout.body if _assigns(x,'input_splits')];outputs=[x for x in layout.body if _assigns(x,'output_splits')]
 assert len(inputs)==len(outputs)==1
 input_device=_download_input(inputs[0].value);output_device=_download_input(outputs[0].value)
 layout.body=[x for x in layout.body if x is not inputs[0] and x is not outputs[0]]
 anchors=[i for i,x in enumerate(layout.body) if _assigns(x,'num_tokens_per_expert')];assert len(anchors)==1
 injected=ast.parse('''
_glm53_host_meta = torch.cat([INPUT_VECTOR, OUTPUT_VECTOR, num_tokens_per_expert]).cpu().numpy().tolist()
assert len(_glm53_host_meta) == 2 * group_size + num_local_experts
input_splits = _glm53_host_meta[:group_size]
output_splits = _glm53_host_meta[group_size:2 * group_size]
_glm53_host_expert_counts = _glm53_host_meta[2 * group_size:]
_glm53_record_metadata_batch(self, len(_glm53_host_meta), num_tokens_per_expert.device.index, topk_idx.shape[0])
''').body
 class Vectors(ast.NodeTransformer):
  def visit_Name(self,node):
   if node.id=='INPUT_VECTOR':return copy.deepcopy(input_device)
   if node.id=='OUTPUT_VECTOR':return copy.deepcopy(output_device)
   return node
 injected=[Vectors().visit(x) for x in injected];at=anchors[0]+1;layout.body[at:at]=injected
 assignments=[i for i,x in enumerate(layout.body) if _is_layout(x)];assert len(assignments)==1
 layout.body[assignments[0]+1:assignments[0]+1]=ast.parse("self._alltoall_layout['_glm53_host_expert_counts'] = _glm53_host_expert_counts").body
 repeats=[]
 for node in ast.walk(layout):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='repeat_interleave':
   assert not any(x.arg=='output_size' for x in node.keywords)
   node.keywords.append(ast.keyword(arg='output_size',value=ast.parse('int(sum(output_splits))',mode='eval').body));repeats.append(node)
 assert len(repeats)==1
 targets=[]
 for node in ast.walk(dispatch):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='tolist':
   device=_download_input(node)
   expected=ast.parse('num_global_tokens_per_local_expert.sum(axis=0)',mode='eval').body
   assert ast.dump(device)==ast.dump(expected),'Unexpected dispatch host metadata calculation'
   targets.append(node)
 assert len(targets)==1
 class CachedCounts(ast.NodeTransformer):
  def visit_Call(self,node):
   if node is targets[0]:return ast.parse("list(self._alltoall_layout['_glm53_host_expert_counts'])",mode='eval').body
   return self.generic_visit(node)
 dispatch=CachedCounts().visit(dispatch)
 namespace={};tree=ast.fix_missing_locations(ast.Module(body=[layout,dispatch],type_ignores=[]))
 module.__dict__['_glm53_record_metadata_batch']=_record
 exec(compile(tree,str(module.__file__)+':'+REVISION,'exec'),module.__dict__,namespace)
 cls._glm53_original_metadata_methods=(original_layout,original_dispatch)
 cls.get_dispatch_layout=namespace['get_dispatch_layout'];cls.dispatch=namespace['dispatch'];cls._glm53_metadata_batch_installed=True
 return {'revision':REVISION,'source_sha256':SOURCE_SHA,'methods':['get_dispatch_layout','dispatch'],'logical_host_vectors':3,'metadata_downloads_before':3,'metadata_downloads_after':1,'repeat_output_size_from_snapshot':True}

def uninstall(module):
 cls=module.AlltoAllNormalCommStrategy;assert cls._glm53_metadata_batch_installed
 cls.get_dispatch_layout,cls.dispatch=cls._glm53_original_metadata_methods;cls._glm53_metadata_batch_installed=False
