"""Read the completed CPU43 formal metrics archive; no model or SDK calls."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261019T-L12D-independent-formal-metrics-archive-v1'
FORMAL='20261019T-L12D-sglang-formal-C32-native-HiCache-v1'
def load(name,file,sha):
    p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def plan():
    old=load('original_native_capture','glm53_sglang_C32_target_graph_native_capture_v3.py','8cad4c626dd24b028d35bdea17863f047d150ac9e15e7e1486669fe5ad668e03')
    mixed=load('raw_safe_capture','glm53_C32_L12C_native_capture_v1.py','e08773c054c1e6a636e9ff6de3e5a0f22257c14ddd6b16e53b0bf36a31d86610')
    client=json.loads((ROOT/'results/sglang-C32-L12D-native-HiCache-v1-client-audit-v1.json').read_text());assert client['client_audit_pass']
    code=old.render(old.CPU_HOST,{'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID,'FORMAL_RID':FORMAL,'CONCURRENCY':32})
    code+="\nassert out['formal_report_identity']['sha256']=="+repr(client['raw_formal_report_sha256'])+"\nassert out['independent_formal_metric_archive_audit']['archive_sha256']=="+repr(client['D_metric_archive_sha256'])+'\n'
    ast.parse(code);return old,mixed,code
def main():
    rpc=json.loads((ROOT/'runs'/FORMAL/'controller-rpc-state.json').read_text());assert rpc['done_received'] and rpc['connection_rc'] is not None
    old,mixed,code=plan();review=json.loads((ROOT/'results/C32-L12D-independent-formal-metrics-actual-review-v1.json').read_text());assert review['host_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'max_SSH':1,'only_completed_CPU_archive_read':True}
    try:
        wire=mixed.invoke_with_raw_failure(old,43,code,'/data/wio/inference-foundry/glm53/runs/'+RID);(run/'node-43.json').write_text(json.dumps(wire,indent=2)+'\n');receipt['node_rc']=wire['rc'];assert wire['rc']==0
        data=json.loads(wire['stdout']);q=data['independent_formal_metric_archive_audit'];assert q['all_native_bodies_SHA_and_DP_gauges_recomputed'] and not q['errors'] and q['sample_pairs']==q['complete_pairs']
        receipt['sample_pairs']=q['sample_pairs'];receipt['native_endpoint_rows']=q['native_HTTP_rows'];receipt['all32productive_Graph_deltas']=len(data['formal_native_target_graph_counters']['delta_by_global_EP_rank'])==32;receipt['pass']=True
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
