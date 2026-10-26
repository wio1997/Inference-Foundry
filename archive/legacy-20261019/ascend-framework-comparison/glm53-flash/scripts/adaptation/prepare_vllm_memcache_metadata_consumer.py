# SPDX-License-Identifier: Apache-2.0
"""Source-only avoid zero-capacity native BM for disabled consumer payloads."""
import argparse,ast,pathlib,hashlib,json,difflib
P=pathlib.Path(__file__).resolve().parents[2];a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a=a.parse_args();a.output.mkdir(parents=True,exist_ok=False)
f=P/'runs/20261017-r124-flash-vllm-performance-preflight/image-sources/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py';original=f.read_text();s=original
x='        self.backend_name = self.backend.lower()\n';y=x+'''        # A consumer with both cache payload directions disabled only needs
        # the native metadata client. MF/MMC1.2 rejects a zero-capacity BM.
        self.memcache_metadata_only_consumer = (
            self.backend_name == "memcache"
            and self.kv_role == "kv_consumer"
            and not extra_config.get("consumer_is_to_load", False)
            and not self.consumer_is_to_put
            and not self.use_layerwise
        )
'''
assert s.count(x)==1;s=s.replace(x,y)
x='        if self.backend_name == "memcache":\n            backend_kwargs["dp_init_barrier"] = memcache_dp_init_barrier\n';y=x+'''            if self.memcache_metadata_only_consumer:
                # Public native backend already supports this for schedulers.
                backend_kwargs["init_bm"] = False
''';assert s.count(x)==1;s=s.replace(x,y)
x='        self.m_store.register_buffer(ptrs, lengths)\n';y='''        if not self.memcache_metadata_only_consumer:
            self.m_store.register_buffer(ptrs, lengths)
''';assert s.count(x)==1;s=s.replace(x,y);ast.parse(s)
(a.output/'pool_worker.py').write_text(s);(a.output/'pool_worker.py.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/pool_worker.py',tofile='b/pool_worker.py')))
manifest={'project_id':'glm53-flash','status':'SOURCE_ONLY_REQUIRES_R140_COMPONENT_AND_FULLMODEL_GATES','base_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest(),'predicate':'backendmemcache+consumer+loadfalse+putfalse+whole-segment only','changed':'NativeMemcacheBackend init_bmFalse and no consumer payloadbufferregistration forpayload-disabledconsumer only. P/allotherbackends/activeconsumerpayloads unchanged. No cachekey/address/block/state/math/SDMA scheduling changes.','runtime_source':'R138Pnativecache put/get602368B exact, DnativeBM0 rejectedmaxMemorySize0. R140actualmetadataSDKcomponent reviewpending.','public_native_API_provenance':'MemcacheBackend supportsinit_bmFalse, nativecreate_scheduler_client alreadyusesit. Wholemodelnativeworker adaptation stillrequires actualDP/Graph/MTP/cache gates.'};(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
