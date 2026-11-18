from pathlib import Path
import hashlib,json

def install(root):
    import sglang
    import sglang.kernels.ops.layernorm.mhc as native
    if getattr(native,'_v41_npu_split_sinkhorn_g353',False):return
    package=Path(sglang.__file__).resolve().parent.parent
    for name,expected in json.loads((root/'native_mhc_sha256.json').read_text()).items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected,name
    from mhc_compat import split_sinkhorn
    prior=native.hc_split_sinkhorn
    def scoped(mixes,scale,base,hc_mult=4,sinkhorn_iters=20,eps=1e-6):
        if mixes.device.type!='npu':return prior(mixes,scale,base,hc_mult,sinkhorn_iters,eps)
        return split_sinkhorn(mixes,scale,base,hc_mult,sinkhorn_iters,eps)
    native.hc_split_sinkhorn=scoped
    native._v41_npu_split_sinkhorn_g353=True
