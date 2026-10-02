"""GLM request tool contract adapter; native reasoning, tool schemas and wire format retained.

Use via vLLM --tool-parser-plugin <absolute file> --tool-call-parser glm47_contract.
Only interpretation changes: absent tools or tool_choice=none keeps tool-marker text
as content rather than inventing a function call. No model/operator work is altered.
"""
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser


class RequestContractGlmParser(Glm47MoeParser):
    def _check_skip_tool_parsing(self, request):
        if not getattr(request, "tools", None) or getattr(request, "tool_choice", None) == "none":
            self.skip_tool_parsing = True
        super()._check_skip_tool_parsing(request)


@ToolParserManager.register_module("glm47_contract")
class Glm47ContractToolParser(Glm47MoeModelToolParser):
    _parser_engine_cls = RequestContractGlmParser
