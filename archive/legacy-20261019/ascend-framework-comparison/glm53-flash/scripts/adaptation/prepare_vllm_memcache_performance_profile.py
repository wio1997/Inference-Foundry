# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned native optimized profile using existing Memcache backend.
No resources/model calls. Component gates are mandatory before deployment.
"""
import argparse,json,pathlib,subprocess,sys,hashlib,tempfile
P=pathlib.Path(__file__).resolve().parents[2]
a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a=a.parse_args();a.output.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix='flash-native-profile-') as td:
 q=pathlib.Path(td)/'base';subprocess.run([sys.executable,str(P/'scripts/adaptation/prepare_vllm_performance_profile.py'),'--output',str(q)],check=True,capture_output=True);profile=json.loads((q/'performance-candidate.json').read_text())
profile['status']='SOURCE_ONLY_NOT_DEPLOYED_OR_QUALIFIED';profile['cache_backend']='Existing pinned memcache-hybrid1.2.0/nativeMemcacheBackend; no Mooncake store module/master';profile['transport']='Native MooncakeConnectorV2 PD via strict MF1.2 SDMA import-only shim, native Memcache AscendStore whole-segment/align. No RDMA fallback.'
profile['component_prerequisite']='R140typedSDMA+Pnativecache/DnativeMetadataonly+documentedSDKcleanup component andactualfullmodelPoolWorker metadataonly adaptation. R138Pcachedata passedbutpairfailedD0BM; do notdeployunpatchedprofile.'
profile['required_source_adaptation']='checks/vllm-native-memcache-metadata-consumer-source-20261017/manifest.json (sourceonly)'
profile['native_Memcache_world_size']='16 Pworkers contribute56GiB each; D16workersmetadata-only/noBM/noDRAM. Not32BMparticipants.'
profile['readiness_gates']=[x for x in profile['readiness_gates'] if not x.startswith(('Resolve actual','R125'))]
profile['readiness_gates'][:0]=['Actual owned ports/backgroundload/npu occupancy recheck; bond4.12 verified. Private Memcache MetaService19052/config19053/metrics19054, fresh own namespace per C.','R127 bulkFP32IEEE component accepted; fullmodel pending, native recurrent/decode/MTP rollback unchanged.','Actual nativeSDK typedSDMA/cache coexistence prereq must pass. Do not deploy if missing.']
for role in profile['roles']:
 p=role['role']=='Prefill';role['pool_segment_size_per_worker']='56GiB requested' if p else '0GiB requested';role['pool_segment_budget_note']='P16 actualworkers x56GiB=896GiB requested plus local buffers; sourceonly /dev/shm capacity headroom101.5GB under R124. NativeMemcache physical allocator/capacity must be observed before C64.'
 for engine in role['engines']:
  argv=engine['argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);extra=kv['kv_connector_extra_config']['connectors'][1]['kv_connector_extra_config'];extra['backend']='memcache';assert extra['use_layerwise'] is False
  if not p:assert extra['consumer_is_to_load'] is extra['consumer_is_to_put'] is False
  argv[i]=json.dumps(kv,separators=(',',':'));env=engine['env'];env.pop('MOONCAKE_CONFIG_PATH');env.update({'MMC_LOCAL_CONFIG_PATH':'/run-evidence/mmc-local-'+('P' if p else 'D')+'.conf','MMC_META_CONFIG_PATH':'/run-evidence/mmc-meta.conf','GLOO_SOCKET_IFNAME':'bond4.12','TP_SOCKET_IFNAME':'bond4.12','HCCL_SOCKET_IFNAME':'bond4.12'});assert env['PYTORCH_NPU_ALLOC_CONF']=='expandable_segments:True' and env['FLASH_SDMA_VERIFY_BYTES']=='0'
 assert role['TP']*role['DP']==16
(a.output/'performance-candidate.json').write_text(json.dumps(profile,indent=2)+'\n')
for role in ['P','D']:
 (a.output/('mmc-local-'+role+'.conf')).write_text('ock.mmc.meta_service_url = tcp://172.16.12.14:19052\nock.mmc.local_service.config_store_url = tcp://172.16.12.14:19053\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 16\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = '+('56GB' if role=='P' else '0GB')+'\n')
(a.output/'mmc-meta.conf').write_bytes((P/'runs/20261017-r134-native-memcache-sdma-coexistence/mmc-meta.conf').read_bytes())
manifest={'project_id':'glm53-flash','status':'SOURCE_ONLY_NOT_DEPLOYABLE_UNTIL_REQUIRED_METADATA_ONLY_ADAPTATION_QUALIFIED','actual_resources_or_model_calls':0,'engines':18,'physical_workers':32,'nativeBM_workers':16,'metadata_only_consumer_workers':16,'selected_backend':'memcache','actual_80000_600_issued':False,'formal_requests':0,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir())}}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
