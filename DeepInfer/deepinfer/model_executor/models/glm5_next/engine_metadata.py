# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-derived engine sizing facts for the bounded GLM text path.

No V4 cache layout, attention preparation, FLOP estimate, model registration,
vision support or runnable engine is implied by this descriptor.
"""
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig


class GLMEngineMetadata:
    def __init__(self, config):
        if not isinstance(config,Glm5NextConfig) or config.quantization_method!='W8A8_DYNAMIC':
            raise ValueError('validated GLM W8A8 checkpoint contract required')
        if config.hf_config['text_config'].get('engram'):
            raise ValueError('GLM engram is not supported by the text path')
        self.config=config
        self.engram_enabled=False
        self.is_vl=False
        self.runtime_ready=False

    @property
    def resolved_tokens_per_block(self):
        # Existing DSA layout and scheduler tests use 128-token logical blocks.
        return 128
