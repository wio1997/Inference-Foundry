# SPDX-License-Identifier: Apache-2.0
"""Check cross-rank setup agreement with mocked collectives, not NPU inference."""
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import patch
import torch
import torch.distributed as dist
from deepinfer.model_executor.models.glm5_next import rank_text_install as module
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next import state_archive


class DenseRankAgreement(TestCase):
    def test_backend_and_limit_must_match_before_install(self):
        mlp = module.GlmDenseMLP(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(),
            hidden_size=4, intermediate_size=8, limit=10.0)
        spec = NS(layer_id=0, mlp="dense", attention="gated_delta_net")
        layer = NS(layer_id=0, attention_kind=spec.attention, mlp=mlp)
        layout = NS(logical_block_size=128, lookahead=0)
        config = NS(layers=[spec], to_dict=lambda: {})
        runner = NS(stack=NS(layers=[layer], config=config, layout=layout))
        actual_tensor = torch.tensor
        peer = []
        def gather(out, local, group=None):
            if not peer:
                peer.append(local.clone())
            out[0].copy_(local)
            out[1].copy_(peer[0])
        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return actual_tensor(*args, **kwargs)
        with patch.object(dist, "is_initialized", return_value=True), \
             patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=gather), \
             patch.object(torch, "tensor", side_effect=cpu_tensor), \
             patch.object(DSACacheLayout, "from_config", return_value=layout), \
             patch.object(state_archive, "_layout_fingerprint", return_value="same-layout"), \
             patch.object(module, "install_rank_dense") as install:
            args = dict(rank=0, world_size=2, cache_bytes=1, max_assembled_bytes=1,
                max_intermediate_rows=8, max_packed_rows=1)
            module.install_rank_text_seams(runner, object(), **args)
            self.assertEqual(install.call_count, 1)
            for backend, limit in (("vllm_native", 10.0), ("torch_fp32", 9.0)):
                mlp.activation_backend, mlp.limit = backend, limit
                with self.assertRaisesRegex(RuntimeError, "structure differs"):
                    module.install_rank_text_seams(runner, object(), **args)
                self.assertEqual(install.call_count, 1)


if __name__ == "__main__":
    main()
