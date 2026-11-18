"""Route V4.1 NPU H5120 to the fixed native HC4 split-hidden Triton kernel."""
import ast,hashlib,inspect,json,textwrap
from pathlib import Path

def install(root):
    import sglang
    import sglang.srt.models.deepseek_v4_mhc as model
    if getattr(model,'_ds41_post_g361',False):return
    package=Path(sglang.__file__).resolve().parent.parent
    for name,expected in json.loads((root/'native_post_sha256.json').read_text()).items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected,name
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    def invoke(x,residual,post_mix,comb):
        return mhc_post_split_h(x.contiguous(),residual.contiguous(),post_mix.contiguous(),comb.contiguous())
    original=model.post
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));fn=tree.body[0]
    indices=[i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='_is_npu'];assert len(indices)==1
    guard=ast.parse("if x.device.type == 'npu' and cfg.mult == 4 and cfg.hidden == 5120:\n    return _ds41_post_g361_call(x, residual, post_mix, comb)").body[0]
    fn.body.insert(indices[0],guard)
    namespace=dict(model.__dict__);exec(compile(ast.fix_missing_locations(tree),str(root/'post_install.py'),'exec'),namespace)
    assert not original.__code__.co_freevars and not namespace['post'].__code__.co_freevars
    model.__dict__['_ds41_post_g361_call']=invoke
    original.__code__=namespace['post'].__code__
    model._ds41_post_g361=True
