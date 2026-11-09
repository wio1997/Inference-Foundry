# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM; modified by DeepSeek.

import torch

from deepinfer.model_executor.layers.tilelang_ops.gather import gather
from deepinfer.model_executor.utils import reconstruct_param


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.nn.Parameter,
    ) -> None:
        super().__init__()

        assert weight.shape == (num_embeddings, embedding_dim)
        self.weight = reconstruct_param(weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = gather(self.weight, input_)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, input_: torch.Tensor) -> None:
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
