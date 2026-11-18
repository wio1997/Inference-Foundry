from pathlib import Path
import hashlib,json

def install(root):
    import sglang
    import sglang.srt.hardware_backend.npu.dsv4.dsv4_rope as native
    if getattr(native, '_v41_fp32_complex_g334', False):
        return
    package=Path(sglang.__file__).resolve().parent.parent
    for name,expected in json.loads((root/'native_rope_sha256.json').read_text()).items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected, name
    from rope_compat import apply_rotary_mul_inplace
    prior=native.Dsv4NpuRoPE.apply_rotary_mul_inplace
    def scoped(q, kv, cos, sin, qk_nope_dim=0):
        if q.device.type != 'npu':
            return prior(q, kv, cos, sin, qk_nope_dim)
        return apply_rotary_mul_inplace(q, kv, cos, sin, qk_nope_dim)
    native.Dsv4NpuRoPE.apply_rotary_mul_inplace=staticmethod(scoped)
    native._v41_fp32_complex_g334=True
