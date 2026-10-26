#!/usr/bin/env python3
"""Prepare reviewed source-only overlays against exact captured rc1 files."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'config/overlays/vllm-sdma-v1'
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'


def sha(data):return hashlib.sha256(data).hexdigest()


def exact_replace(text, old, new):
    if text.count(old)!=1:raise RuntimeError('Source anchor not unique: '+old[:90])
    return text.replace(old,new,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    captures=[ROOT/'runs/20261017T0205-inventory-compact/source-30',
              ROOT/'runs/20261017T0300-framework-source/code-30']
    records={}
    for directory in captures:
        for entry in json.loads((directory/'index.json').read_text()):
            records[entry['path']]={**entry,'capture':directory/entry['local']}
    prefix='/vllm-workspace/vllm-ascend/vllm_ascend/'
    changes=[]
    def edit(relative, transform):
        entry=records[prefix+relative];data=entry['capture'].read_bytes()
        if sha(data)!=entry['sha256']:raise RuntimeError('Captured source SHA mismatch')
        original=data.decode();modified=transform(original);ast.parse(modified)
        target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(modified)
        patch=OUT/'patches'/(relative.replace('/','__')+'.patch');patch.parent.mkdir(exist_ok=True)
        patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=relative,tofile=relative)))
        changes.append({'container_path':prefix+relative,'local':str(target.relative_to(OUT)),
                        'original_sha256':sha(data),'patched_sha256':sha(modified.encode()),
                        'patch':str(patch.relative_to(OUT))})
    def base_worker(text):
        text=exact_replace(text,
            'from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (\n    global_te,\n)',
            'from glm53_sdma_transport import global_te')
        text=exact_replace(text,'        self.engine = global_te.get_transfer_engine(',
            '''        sdma_cfg = self.kv_transfer_config.kv_connector_extra_config.get("glm53_sdma")
        if not isinstance(sdma_cfg, dict) or set(sdma_cfg) != {"store_url", "session_port_offset", "sdk_log_level"}:
            raise ValueError("This GLM53 overlay requires its explicit SDMA-only configuration")
        global_te.configure(
            role="Prefill" if self.kv_transfer_config.is_kv_producer else "Decode",
            device_id=int(torch.npu.current_device()),
            session_port=self.handshake_port + int(sdma_cfg["session_port_offset"]),
            store_url=sdma_cfg["store_url"],
            sdk_log_level=int(sdma_cfg["sdk_log_level"]),
        )
        self.engine = global_te.get_transfer_engine(''')
        return exact_replace(text,'        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)',
            '        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)\n        global_te.pin_kv_caches(kv_caches)')
    edit('distributed/kv_transfer/kv_p2p/mooncake/base_worker.py',base_worker)
    def pull_worker(text):
        text=exact_replace(text,'            remote_engine_id, requests = self.request_queue.get()',
            '''            remote_engine_id, requests = self.request_queue.get()
            if remote_engine_id is None and requests is None:
                self.request_queue.task_done()
                self.executor.shutdown(wait=True)
                return''')
        text=exact_replace(text,'        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)',
            '''        from glm53_sdma_transport import audit
        state_specs = []
        for spec_index, layers in transfer_entries_by_spec.items():
            spec = self.kv_cache_specs[spec_index]
            state_specs.append({
                "spec_index": spec_index, "class": type(spec).__name__,
                "layers": sorted({self.layer_names[local] for local, _ in layers}),
                "dtype": str(getattr(spec, "dtype", None)),
                "scale_dtype": str(getattr(spec, "scale_dtype", None)),
                "page_bytes": getattr(spec, "page_size_bytes", None),
            })
        audit("model_state_read_submit", request_ids=request_ids, remote_session=session_id,
              state_specs=state_specs, payload_bytes_requested=sum(length_list),
              ranges=len(length_list), unique_ranges=len(set(zip(src_list, dst_list, length_list))))
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        audit("model_state_read_complete", request_ids=request_ids, remote_session=session_id,
              rc=ret, payload_bytes_completed=sum(length_list) if ret == 0 else None)''')
        return text
    edit('distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py',pull_worker)
    def cache_thread(text):
        text=exact_replace(text,'                request_data = self.request_queue.get()',
            '''                request_data = self.request_queue.get()
                from glm53_runtime_lifecycle import STOP_CACHE_THREAD
                if request_data is STOP_CACHE_THREAD:
                    self.request_queue.task_done()
                    return''')
        return text
    edit('distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py',cache_thread)
    def multi(text):
        return exact_replace(text,'    def configure_dspark_prefix_cache(self, vllm_config: "VllmConfig") -> bool:',
            '''    def shutdown(self):
        from glm53_runtime_lifecycle import shutdown_multi
        shutdown_multi(self)

    def configure_dspark_prefix_cache(self, vllm_config: "VllmConfig") -> bool:''')
    edit('distributed/kv_transfer/ascend_multi_connector.py',multi)
    helpers=[]
    for name in ['glm53_sdma_transport.py','glm53_runtime_lifecycle.py']:
        data=(ROOT/'scripts'/name).read_bytes();ast.parse(data.decode())
        dest=OUT/'python'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
        helpers.append({'local':str(dest.relative_to(OUT)),'sha256':sha(data)})
    manifest={'project_id':'glm53','image_id':IMAGE,'status':'PREPARED_NOT_DEPLOYED',
              'required_sdk_archive_sha256':'f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3',
              'files':changes,'helpers':helpers,'scope':'SDMA-only PD adapter, per-request state/byte audit, bounded native SDK shutdown; no performance optimization claimed'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'overlay':str(OUT),'files':len(changes),'helpers':len(helpers)}))

if __name__=='__main__':main()
