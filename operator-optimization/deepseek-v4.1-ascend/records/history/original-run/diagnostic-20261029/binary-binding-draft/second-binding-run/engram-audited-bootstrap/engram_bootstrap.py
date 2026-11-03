"""Opt-in, narrowly scoped bootstrap: no model import, no NPU operations."""
import hashlib, importlib.abc, importlib.machinery, importlib.util, inspect, json, os, sys, types
TARGET='vllm_ascend.models.deepseek_v41.engram.embedding'
CLASS='AscendParallelEngramEmbedding'
EXPECTED_FILE='/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/embedding.py'
EXPECTED_SHA='a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9'

def _find_code(code,qualname):
    matches=[]
    def walk(c,prefix=''):
        for x in c.co_consts:
            if isinstance(x,types.CodeType):
                path=(prefix+'.' if prefix else '')+x.co_name
                if path==qualname:matches.append(x)
                walk(x,path)
    walk(code)
    return matches

def _canonical(value,depth=0):
    if depth>64:raise ValueError('code nesting')
    if isinstance(value,types.CodeType):
        fields=('co_code','co_consts','co_names','co_varnames','co_filename','co_name','co_firstlineno','co_flags','co_stacksize','co_argcount','co_kwonlyargcount','co_posonlyargcount','co_freevars','co_cellvars','co_nlocals')
        for optional in ('co_qualname','co_linetable','co_lnotab','co_exceptiontable'):
            if hasattr(value,optional):fields+=(optional,)
        return {'type':'code','fields':{n:_canonical(getattr(value,n),depth+1) for n in fields}}
    if value is None:return {'type':'none'}
    if value is Ellipsis:return {'type':'ellipsis'}
    if type(value) in (bool,int,str):return {'type':type(value).__name__,'value':value}
    if type(value) is float:return {'type':'float','value':value.hex()}
    if type(value) is complex:return {'type':'complex','value':[value.real.hex(),value.imag.hex()]}
    if type(value) is bytes:return {'type':'bytes','value':value.hex()}
    if type(value) is tuple:return {'type':'tuple','value':[_canonical(x,depth+1) for x in value]}
    if type(value) is frozenset:
        items=[_canonical(x,depth+1) for x in value]
        return {'type':'frozenset','value':sorted(items,key=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')))}
    raise ValueError('unsupported code constant')

def _code_digest(code):
    raw=json.dumps(_canonical(code),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()

def _patch(module):
    try:
        spec=getattr(module,'__spec__',None)
        if not spec or spec.origin!=EXPECTED_FILE:raise ValueError('origin')
        with open(EXPECTED_FILE,'rb') as f:source=f.read(4*1024*1024+1)
        if len(source)>4*1024*1024 or hashlib.sha256(source).hexdigest()!=EXPECTED_SHA:raise ValueError('source hash')
        cls=getattr(module,CLASS,None)
        if not isinstance(cls,type) or cls.__module__!=TARGET:raise ValueError('class')
        fn=cls.__dict__.get('embed_gathered')
        if not isinstance(fn,types.FunctionType):raise ValueError('function')
        params=inspect.signature(fn).parameters
        if list(params)!=['self','gathered','num_tokens']:raise ValueError('signature')
        codes=_find_code(compile(source,EXPECTED_FILE,'exec',dont_inherit=True,optimize=sys.flags.optimize),CLASS+'.embed_gathered')
        if len(codes)!=1 or _code_digest(codes[0])!=_code_digest(fn.__code__):raise ValueError('code mismatch')
        import audited_observer
        facts={'status':'verified_before_install','source_sha256':EXPECTED_SHA,'code_sha256':_code_digest(fn.__code__),'class':CLASS}
        if not audited_observer.install(cls,facts):raise ValueError('install failed')
        wrapped=cls.__dict__.get('embed_gathered')
        if not getattr(wrapped,'_wio_engram_metadata',False) or getattr(wrapped,'__wrapped__',None) is not fn:raise ValueError('wrapper mismatch')
    except Exception:
        # Missing status = unknown to controller; never break original import.
        return False
    return True

class _Loader(importlib.abc.Loader):
    def __init__(self,inner):self.inner=inner
    def create_module(self,spec):
        fn=getattr(self.inner,'create_module',None)
        return fn(spec) if fn else None
    def exec_module(self,module):
        self.inner.exec_module(module)
        _patch(module)
    def __getattr__(self,name):return getattr(self.inner,name)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        # Leave unusual loaders untouched; coverage then remains unknown.
        if spec and spec.origin==EXPECTED_FILE and isinstance(spec.loader,importlib.machinery.SourceFileLoader):spec.loader=_Loader(spec.loader)
        return spec

def activate():
    if os.environ.get('WIO_ENGRAM_META_ENABLE')!='1':return False
    if TARGET in sys.modules or any(isinstance(x,_Finder) for x in sys.meta_path):return False
    for i,x in enumerate(sys.meta_path):
        if x is importlib.machinery.PathFinder:
            sys.meta_path.insert(i,_Finder());return True
    return False
