#!/usr/bin/env python3
"""History adoption plus one verified-at-D proposal from existing P MTP1 work."""
import argparse
import ast
import base64
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib


def sha(data):return hashlib.sha256(data).hexdigest()

def once(s,old,new):
    assert s.count(old)==1,repr(old)
    return s.replace(old,new,1)


SEED_CHECK='''    def _glm53_valid_p_draft(self, request):
        params = request.kv_transfer_params or {}
        packet = params.get("p_first_draft")
        return bool(isinstance(packet, dict)
            and packet.get("rank_agreement") == 16
            and packet.get("request_id") == params.get("remote_request_id")
            and packet.get("num_prompt_tokens") == request.num_prompt_tokens
            and packet.get("first_token_id") == params.get("last_token_id")
            and isinstance(packet.get("token_ids"), list) and len(packet["token_ids"]) == 1
            and type(packet["token_ids"][0]) is int
            and 0 <= packet["token_ids"][0] < self.vllm_config.model_config.get_vocab_size())

'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--divergence-probe',action='store_true');p.add_argument('--shared-overlap-probe',action='store_true');p.add_argument('--unused-single-token-probe',action='store_true');p.add_argument('--attention-state-probe',action='store_true');p.add_argument('--prefill-padding-probe',action='store_true');p.add_argument('--prefill-attention-probe',action='store_true');p.add_argument('--sfa-replay-order-probe',action='store_true');p.add_argument('--mtp-ffn-graph-probe',action='store_true');p.add_argument('--mtp-moe-graph-probe',action='store_true');p.add_argument('--rope-cache-layout-probe',action='store_true');p.add_argument('--moe-prefix-histogram-probe',action='store_true');p.add_argument('--moe-cumsum-i32-probe',action='store_true');p.add_argument('--moe-prefix-gmm2-count-probe',action='store_true');a=p.parse_args();assert not (a.shared_overlap_probe or a.unused_single_token_probe or a.attention_state_probe) or a.divergence_probe
    assert not a.output.exists();root=Path(__file__).resolve().parent;parent=root.parent/'p-first-adopt-v1'
    spec=importlib.util.spec_from_file_location('adopt_build',parent/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    pins={**base.PINS,'utils.py':('vllm/vllm/distributed/kv_transfer/kv_connector/utils.py','317ed40dc837671ca6e666165222e641a0f9b3992aea63bbf54de7e13d40b3a4')}
    native={name:(a.source_root/path).read_bytes() for name,(path,_) in pins.items()}
    for name,(_,expected) in pins.items():assert sha(native[name])==expected,name
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)/'v1'
        subprocess.run([sys.executable,str(parent/'build.py'),'--source-root',str(a.source_root),'--output',str(tmp)],check=True,stdout=subprocess.DEVNULL)
        contents={name:(tmp/name).read_text() if name in base.PINS else native[name].decode() for name in pins}
    s=contents['scheduler.py']
    s=once(s,'        self._glm53_adopt_pending = {}\n','        self._glm53_adopt_pending = {}\n        self._glm53_export_p_draft_enabled = os.environ.get("GLM53_P_FIRST_DRAFT_EXPORT") == "1"\n')
    s=once(s,'    def _glm53_can_adopt_first(self, request):\n',SEED_CHECK+'    def _glm53_can_adopt_first(self, request):\n')
    s=once(s,'            self._glm53_adopt_enabled and params.get("_glm53_adopt_first") is True\n','            self._glm53_adopt_enabled and params.get("_glm53_adopt_first") is True\n            and (params.get("_glm53_use_p_draft") is not True or self._glm53_valid_p_draft(request))\n')
    s=once(s,'                request._glm53_adopted = True\n','                request._glm53_adopted = True\n                request._glm53_initial_seed = (list(request.kv_transfer_params["p_first_draft"]["token_ids"])\n                    if request.kv_transfer_params.get("_glm53_use_p_draft") is True else [])\n')
    s=once(s,'''                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
''','''                    initial_seed = getattr(request, "_glm53_initial_seed", [])
                    assert len(initial_seed) <= self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = initial_seed + [-1] * (self.num_spec_tokens - len(initial_seed))
                    if initial_seed:
                        logger.info("[glm53/p-first-seed] request=%s seed=%s placeholders=%s", request_id, initial_seed, self.num_spec_tokens - len(initial_seed))
                request._glm53_initial_seed = []
''')
    s=once(s,'            if should_emit_output:\n                # Add EngineCoreOutput for this Request.\n','''            if should_emit_output:
                if kv_transfer_params is not None and self._glm53_export_p_draft_enabled:
                    ids = getattr(model_runner_output, "spec_token_ids", None)
                    idx = model_runner_output.req_id_to_index.get(req_id)
                    if ids is not None and idx is not None and len(ids[idx]) == 1:
                        kv_transfer_params["p_first_draft"] = {
                            "token_ids": list(ids[idx]), "request_id": req_id,
                            "num_prompt_tokens": request.num_prompt_tokens,
                            "first_token_id": request.output_token_ids[0], "rank_agreement": 16}
                # Add EngineCoreOutput for this Request.
''')
    contents['scheduler.py']=s
    s=contents['output.py']
    s=once(s,'    initial_output_token_ids: list[int] | None = None\n','    initial_output_token_ids: list[int] | None = None\n    export_p_first_draft: bool = False\n')
    s=once(s,'            initial_output_token_ids=(list(request.output_token_ids) if getattr(request, "_glm53_adopted", False) else None),\n','            initial_output_token_ids=(list(request.output_token_ids) if getattr(request, "_glm53_adopted", False) else None),\n            export_p_first_draft=(request.kv_transfer_params or {}).get("_glm53_export_p_draft") is True,\n')
    contents['output.py']=s
    s=contents['utils.py']
    s=once(s,'        assert output is not None\n        output.kv_connector_output = KVConnectorOutput(\n','''        assert output is not None
        if __import__("os").environ.get("GLM53_P_FIRST_DRAFT_EXPORT") == "1":
            proposed = getattr(output, "spec_token_ids", None)
            if proposed is not None:
                valid = (len(outputs) == 16 and len(proposed) == len(output.req_ids)
                    and all(item is not None and item.req_ids == output.req_ids
                        and getattr(item, "spec_token_ids", None) == proposed
                        and item.sampled_token_ids == output.sampled_token_ids for item in outputs)
                    and all(not row or (len(row) == 1 and type(row[0]) is int and row[0] >= 0) for row in proposed))
                if not valid or not any(proposed):
                    output.spec_token_ids = None
        output.kv_connector_output = KVConnectorOutput(
''')
    contents['utils.py']=s
    # Rebuild only the appended helpers; every native NPU function stays intact.
    s=native['model_runner_v1.py'].decode()
    helpers=[('trace.py','GLM53_P_FIRST_ADOPT_TRACE','trace'),('producer.py','GLM53_P_FIRST_DRAFT_EXPORT','export')]
    if a.attention_state_probe:helpers.append(('attention_snapshot.py','GLM53_SFA_STATE_SNAPSHOT','attention'))
    if a.divergence_probe:helpers.append(('divergence.py','GLM53_FIRST_DIVERGENCE_PROBE','divergence'))
    if a.shared_overlap_probe:helpers.append(('shared_overlap_probe.py','GLM53_SHARED_OVERLAP_DIAGNOSTIC','overlap'))
    if a.unused_single_token_probe:helpers.append(('unused_single_token_draft.py','GLM53_UNUSED_SINGLE_TOKEN_DRAFT_PROBE','unused'))
    if a.prefill_padding_probe:helpers.append(('prefill_padding.py','GLM53_PREFILL_PADDING_PROBE','prefill_padding'))
    if a.prefill_attention_probe:helpers.append(('prefill_attention.py','GLM53_PREFILL_ATTENTION_PROBE','prefill_attention'))
    if a.sfa_replay_order_probe:helpers.append(('sfa_replay_order.py','GLM53_SFA_REPLAY_ORDER_PROBE','sfa_replay_order'))
    if a.mtp_ffn_graph_probe:helpers.append(('mtp_ffn_graph.py','GLM53_MTP_FFN_GRAPH_PROBE','mtp_ffn_graph'))
    if a.mtp_moe_graph_probe:helpers.append(('mtp_moe_graph.py','GLM53_MTP_MOE_GRAPH_PROBE','mtp_moe_graph'))
    if a.rope_cache_layout_probe:helpers.append(('rope_cache_layout.py','GLM53_ROPE_CACHE_LAYOUT_PROBE','rope_cache_layout'))
    assert sum([a.moe_prefix_histogram_probe, a.moe_cumsum_i32_probe, a.moe_prefix_gmm2_count_probe]) <= 1
    if a.moe_cumsum_i32_probe:helpers.append(('moe_cumsum_i32.py','GLM53_MOE_CUMSUM_I32_PROBE','moe_cumsum_i32'))
    if a.moe_prefix_gmm2_count_probe:helpers.append(('moe_prefix_gmm2_count.py','GLM53_MOE_PREFIX_GMM2_COUNT_PROBE','moe_prefix_gmm2_count'))
    if a.moe_prefix_histogram_probe:helpers.append(('moe_prefix_histogram.py','GLM53_MOE_PREFIX_HISTOGRAM_PROBE','moe_prefix_histogram'))
    for filename,gate,name in helpers:
        helper=(root/filename).read_text();ast.parse(helper)
        if filename in ['moe_cumsum_i32.py', 'moe_prefix_gmm2_count.py']:
            base=(root/'moe_prefix_histogram.py').read_text();assert sha(base.encode())=='173c720dd79dd656ec9cd88b51eeb373752ca71567254f88bcb1d367fcfac90c'
            helper='_BASE_PREFIX_SOURCE='+repr(base)+'\n'+helper
        s+='\n\nif __import__(\'os\').environ.get('+repr(gate)+") == '1':\n    import types as _glm_types\n    _glm_"+name+" = _glm_types.ModuleType('_glm_seed_"+name+"')\n    exec(compile("+repr(helper)+", '<glm53-seed-"+name+">', 'exec'), _glm_"+name+".__dict__)\n    _glm_"+name+".install(globals())\n"
    contents['model_runner_v1.py']=s
    for name,text in contents.items():ast.parse(text)
    identity=sha(''.join(contents[n] for n in sorted(contents)).encode());remote='/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/p-first-seed-v2/'+identity[:12]
    a.output.mkdir(parents=True);patches=[];payload={}
    for name,text in contents.items():
        data=text.encode();(a.output/name).write_bytes(data);payload[name]=base64.b64encode(zlib.compress(data)).decode()
        (a.output/(name+'.patch')).write_text(''.join(difflib.unified_diff(native[name].decode().splitlines(True),text.splitlines(True),fromfile=name,tofile=name)))
        patches.append({'source':remote+'/'+name,'target':'/vllm-workspace/'+pins[name][0],'sha256':sha(data),'base_sha256':pins[name][1]})
    stager='''import base64,hashlib,json,socket,zlib
from pathlib import Path
assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
root=Path(ROOT);root.mkdir(parents=True,exist_ok=True)
for name,value in PAYLOAD.items():
 data=zlib.decompress(base64.b64decode(value));assert hashlib.sha256(data).hexdigest()==HASHES[name]
 dest=root/name
 if dest.exists():assert dest.read_bytes()==data
 else:
  with dest.open('xb') as out:out.write(data)
  dest.chmod(0o444)
print('GLM53_RESULT_JSON_BEGIN\\n'+json.dumps({'staged':True,'root':str(root),'files':HASHES,'device_ops':False})+'\\nGLM53_RESULT_JSON_END')
'''.replace('ROOT',repr(remote)).replace('PAYLOAD',repr(payload)).replace('HASHES',repr({n:sha(s.encode()) for n,s in contents.items()}))
    ast.parse(stager);(a.output/'stage_overlay.py').write_text(stager)
    manifest={'identity':identity,'patches':patches,'remote_root':remote,'stage_sha256':sha(stager.encode()),'P_K':1,'D_K':5,'draft_confirmed_output':False,'history':'only confirmed y1','computed':'N','initial_spec_slots':'one unverified P proposal then four native invalid placeholders','new_device_copy':'tiny pinned CPU copy on MTP writer stream before original Async dependency; SDMA profiling pending','default_adoption':False,'device_validated':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
