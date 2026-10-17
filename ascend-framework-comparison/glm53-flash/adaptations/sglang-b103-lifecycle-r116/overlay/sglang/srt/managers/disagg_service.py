"""Start bootstrap/kv-store-related server"""

import logging
import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_serving,
)

logger = logging.getLogger(__name__)


def start_disagg_service():
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(get_disagg().disaggregation_mode)
    transfer_backend = TransferBackend(get_disagg().disaggregation_transfer_backend)

    # With role switching, run bootstrap on every instance (not just prefill) so
    # one flipped to prefill already has it; it isn't rebuilt on flip.
    start_bootstrap = disagg_mode == DisaggregationMode.PREFILL or (
        get_disagg().enable_pd_role_switch and disagg_mode != DisaggregationMode.NULL
    )

    if start_bootstrap and get_disagg().enable_pd_role_switch:
        logger.warning(
            "Role switch starts a bootstrap server on this instance at %s:%d. "
            "If another PD instance runs on the same host, give each one a "
            "distinct --disaggregation-bootstrap-port or the bind will conflict.",
            get_serving().host,
            get_disagg().disaggregation_bootstrap_port,
        )

    if start_bootstrap:
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=get_serving().host,
            port=get_disagg().disaggregation_bootstrap_port,
        )
        maybe_create_ascend_config_store(transfer_backend=transfer_backend)

        return bootstrap_server


def maybe_create_ascend_config_store(transfer_backend: TransferBackend) -> None:
    """Also called directly by the rust-server scheduler: there the KV
    bootstrap registry is served by the embedded rust server's api listener
    (one rust implementation covers every transfer backend — their
    bootstrap-server subclasses are all plain ``CommonKVBootstrapServer``,
    which the rust registry ports verbatim), leaving this store as the only
    ``start_disagg_service`` duty left to perform."""
    if not (
        get_parallel().node_rank == 0 and transfer_backend == TransferBackend.ASCEND
    ):
        return
    try:
        from memfabric_hybrid import create_config_store, set_conf_store_tls, set_log_level
        import pathlib, hashlib
        from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
        if os.environ.get('FLASH_SDMA_VERIFY_BYTES')!='1':
            raise ValueError('This Flash diagnostic requires mandatory byte proof')
        ascend_url = os.environ['ASCEND_MF_STORE_URL']
        if ascend_url!='tcp://172.16.12.46:19024':
            raise ValueError('Unexpected Flash native Prefill control-store URL')
        # Same effective order as the accepted R089 native-function fixture.
        set_conf_store_tls(False, '')
        set_log_level(0)
        ret = create_config_store(ascend_url)
        append('native_control_store',store_url=ascend_url,ret=ret,
               explicit_TLS=False,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
        if ret!=0:raise RuntimeError(f'Flash native config-store failed: {ret}')
    except Exception as e:
        raise RuntimeError(
            f"Failed create mf store, invalid ascend_url. With exception {e}"
        )
