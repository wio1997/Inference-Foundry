#!/usr/bin/env python3
"""Pin and read missing, explicitly named head/request contract dependencies."""
from pathlib import Path
import importlib.util

path = Path(__file__).with_name('collect_frozen_sources.py')
spec = importlib.util.spec_from_file_location('source_transport', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = module.REMOTE
begin = source.index('selection={')
end = source.index("manifest={'scope':", begin)
selection = {
    'vllm-ascend': [],
    'vllm': [
        'vllm/__init__.py',
        'vllm/model_executor/models/qwen3_dspark.py',
        'vllm/model_executor/models/qwen3_dflash.py',
        'vllm/model_executor/models/utils.py',
        'vllm/model_executor/models/interfaces.py',
        'vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py',
        'vllm/sampling_params.py',
        'vllm/config/model.py',
        'vllm/entrypoints/openai/engine/serving.py',
        'vllm/entrypoints/openai/engine/protocol.py',
    ],
}
module.REMOTE = source[:begin] + 'selection=' + repr(selection) + '\n' + source[end:]
work = Path(__file__).resolve().parent / 'contract-source-supplement'
work.mkdir(exist_ok=False)
module.__file__ = str(work / 'collect_frozen_sources.py')
module.main()
