"""Independent opt-in bootstrap; no torch/model import at startup."""
import os
if os.environ.get('WIO_ENGRAM_META_ENABLE')=='1':
    try:
        import engram_bootstrap
        engram_bootstrap.activate()
    except Exception:
        pass
