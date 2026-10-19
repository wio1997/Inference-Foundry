"""Local config preparation only; no resource access or runtime mutation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(fw,parent,dest,nodes,head,max_gb):
    out=ROOT/dest;assert not out.exists();out.mkdir();parents={};files={}
    for node in nodes:
        p=ROOT/parent/f'node-{node}.json';parents[str(p.relative_to(ROOT))]=sha(p);j=json.loads(p.read_text())
        j['phase']='OFFICIAL_STANDALONE_MC_BUDGET_LEASE_NATIVE_OBSERVABILITY_QUALIFICATION_NOT_BASELINE'
        j['environment_intent']={**j.get('environment_intent',{}),'PYTHONHASHSEED':'0','ACL_OP_INIT_MODE':'1','ASCEND_ENABLE_USE_FABRIC_MEM':'1'}
        j['required_meta_service_lease_ttl_ms']=60000
        j['standalone_pool_intent']={'provider_nodes':nodes[:2],'provider_processes_per_node':1,'provider_device_id':0,'provider_dram_GiB_each':max_gb,'model_client_dram_GiB':0,'all_local_processes_max_dram_GiB':max_gb,'world_size':128,'init_bm':True,'must_start_provider_before_weights':True,'actual_allocation_pending':True,'SDK_reference':'SDK1.3configuration +vLLMrc1KVPoolStep6'}
        if fw=='sglang':
            j.pop('host_L2_counterfactual',None);j.pop('source_failure_evidence',None)
            args=j['launch_args']
            if j['role']=='P':
                i=args.index('--hicache-size');assert args[i+1]=='10';args[i+1]='9'
                i=args.index('--hicache-storage-backend-extra-config');extra=json.loads(args[i+1]);assert extra['dram_size']=='16GB'
                extra.update(dram_size='0GB',max_dram_size='368GB',hbm_size='0GB',max_hbm_size='0GB');args[i+1]=json.dumps(extra)
            j['cache_budget_intent']={'P_provider_MC_GiB':736,'P_hicache_size_decimal_GB_each':9,'P_host_tokens_native_AST_expected':81152,'P_host_data_buffers_bytes_AST_expected':288189579264,'P_host_staging_GiB_AST_expected':268.3974609375,'D_persistent_HiCache_data_buffers_bytes':0,'P_host_buffers_are_retained_L2_cache':False,'MC_plus_staging_GiB_AST_expected':1004.3974609375,'headroom_to1TiB_GiB_before_overheads':19.6025390625,'evidence':'results/sglang-native-staging9-budget-fixture-v1.json','actual_new_allocation_and_retention_pending':True}
            review=j['native_buffer_only_review'];review['compatibility_pending']=False
            review['single_launch_change']='Previousbufferonly10GBqualified; newstaging9GB+standalone0GBclient/max368GB/nativelease60s/documentedenv requirefreshqualification.'
            review['functional_evidence']='results/sglang-buffer-only-cache-qualification-v1.json'
            review['constraints']='FULLtree/nativewrite_through/whole80Kbouncesize81152viaAST. No retainedL2; stagingfreedatstorageack. Actualnewallocation/readbackpending.'
            j['moe_dispatch_intent']['long_cache_qualification_pending']=False
            j['moe_dispatch_intent']['model_evidence']=['results/sglang-pd-sdma-matrix-v3.json','results/sglang-buffer-only-cache-qualification-v1.json']
            j['hicache_host_io_intent']['actual_copy_pending']=False
        p=out/f'node-{node}.json';p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');files[p.name]=sha(p)
    port=29680 if fw=='vllm' else 29780
    meta=f'ock.mmc.meta_service_url = tcp://172.16.12.{head}:{port}\nock.mmc.meta_service.config_store_url = tcp://172.16.12.{head}:{port+1}\nock.mmc.meta_service.metrics_url = http://172.16.12.{head}:{port+2}\nock.mmc.log_level = info\nock.mmc.meta.lease_ttl_ms = 60000\nock.mmc.evict_threshold_high = 90\nock.mmc.evict_threshold_low = 89\n'
    local=f'ock.mmc.meta_service_url = tcp://172.16.12.{head}:{port}\nock.mmc.local_service.config_store_url = tcp://172.16.12.{head}:{port+1}\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 128\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = 0GB\nock.mmc.local_service.max.dram.size = {max_gb}GB\nock.mmc.local_service.hbm.size = 0GB\nock.mmc.local_service.max.hbm.size = 0GB\n'
    for name,text in [('meta.conf',meta),('local.conf',local)]:
        p=out/name;p.write_text(text);files[name]=sha(p)
    review={'phase':'LOCAL_PREP_ONLY_NOT_STAGED_OR_STARTED','framework':fw,'parent_files_sha256':parents,'files_sha256':files,'official_docs_index':'runs/20261017T1340-official-memcache-docs-review/index.json','common_native_Meta_policy':{'lease_ttl_ms':60000,'evict_high':90,'evict_low':89,'rationale':'Supportednativelease/samebothframeworks. Low89aboveSGC64active637.55/736=86.62pct limitsoldrunLRUhysteresis; actualallocation/retentionpending, noframeworkcachepatch.'},'runtime_requirements':['Freshnativearg/sourcepreflight +officialphysicalmemoryscan beforeproviderallocation','Actualpool/capacity/wholeprefixwrite/readback andP4D16SDMA/MTP beforeformal','OwnproviderPIDexplicitwhitelist, no existingstarterempty-NPUassumption','No manualcacheclear; newCnamespaceensuresfreshsuffixes, observe naturalLRUretention','vLLMnativeprofiler remainsinactiveuntilafterformalbaseline']}
    (out/'manifest.json').write_text(json.dumps(review,indent=2)+'\n');print(dest,sha(out/'manifest.json'))

if __name__=='__main__':
    prepare('vllm','config/vllm-correctness-v5','config/vllm-correctness-v6',[30,31,43,45],30,512)
    prepare('sglang','config/sglang-correctness-v7','config/sglang-correctness-v8',[34,35,36,41],34,368)
