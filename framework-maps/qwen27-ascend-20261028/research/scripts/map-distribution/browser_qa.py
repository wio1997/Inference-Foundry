"""Check real native selection and retained evidence; default requires actual204."""
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import argparse,hashlib,json,threading,time
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic-single-rank',action='store_true');p.add_argument('--mode',choices=['both','HTTP','file'],default='both');a=p.parse_args()
M=a.map.resolve();O=a.output.resolve();assert not O.exists();O.mkdir(parents=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
b=json.loads((M/'combined-map.json').read_text());nm=b['new_measurements'];ix=nm['native_sharded_map']['index'];expected=1 if a.diagnostic_single_rank else 204
assert ix['identity_count']==len(ix['per_rank'])==expected and ix['bundle_errors']==0
assert len({r['rank_key']for r in ix['per_rank']})==expected
if not a.diagnostic_single_rank:
 assert {r['framework']for r in ix['per_rank']}=={'vllm','sglang'}
 c=json.loads((M/'composition-receipt.json').read_text());assert c['verified_identity_count']==204 and c['output_combined_map']['sha256']==sha((M/'combined-map.json').read_bytes())
assert nm['postfreeze_root_audit']['vllm_tasktag_candidate_membership']['summary']['accepted_completion_fence_count']==0
# One real rank per framework, plus the largest inflated rank to exercise memory-sensitive UI.
selected={}
for row in ix['per_rank']:selected.setdefault(row['framework'],row)
largest=max(ix['per_rank'],key=lambda r:r['detail']['json_bytes']);selected[largest['rank_key']]=largest
selected={r['rank_key']:r for r in selected.values()}
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(M)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
errors=[];console=[];tests=[];contract_tests=[]
try:
 with sync_playwright()as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport={'width':1440,'height':1000})
  page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:console.append(m.text)if m.type=='error'else None)
  for mode,url in [('file',(M/'index.html').as_uri()),('HTTP',f'http://127.0.0.1:{server.server_port}/index.html')]:
   if a.mode!='both' and mode!=a.mode:continue
   page.set_viewport_size({'width':1440,'height':1000})
   page.goto(url,wait_until='load',timeout=90000)
   page.wait_for_function("document.getElementById('timeline-content').textContent.includes('Shard SHA verified')",timeout=60000)
   assert '质量' in page.locator('#baseline-content').inner_text() or 'quality' in page.locator('#baseline-content').inner_text().lower()
   page.locator('[data-tab="profiles"]').click()
   for row in selected.values():
    page.locator('#native-shard-framework').select_option(row['framework']);page.locator('#native-shard-label').select_option(row['label']);page.locator('#native-shard-rank').select_option(row['rank_key'])
    page.wait_for_function("r=>{const root=document.getElementById('native-shard-framework').closest('article')||document.getElementById('native-shard-framework').parentElement.parentElement;const t=root.textContent;if(t.includes('Selected rank shard failed'))throw Error(t);return t.includes(r.framework+' · '+r.label+' · '+r.rank_basename+' · explicit rank')}",arg=row,timeout=120000)
    sections=[]
    for sec in ['task_groups','profile_rows','layer_rows','step_native_rows','eager_audits','eager_layer_rows']:
     page.locator('#native-shard-section').select_option(sec)
     result=page.evaluate('''({sec,r})=>{const ck=(x,m)=>{if(!x)throw Error(m)};const root=document.getElementById('native-shard-section').parentElement.parentElement;const h=[...root.querySelectorAll('h3')].at(-1).textContent;const n=Number(h.match(/· (\d+) rows$/)?.[1]);ck(Number.isSafeInteger(n),'row heading');if(sec!=='task_groups'&&r.row_counts[sec]!==null)ck(n===r.row_counts[sec],'index row counts');ck(root.textContent.includes('Missing source values remain unknown'),'unknown scope');ck(root.textContent.includes('no rank/PID durations are added'),'no crossrank sum');return{section:sec,rendered_count:n,index_count:r.row_counts[sec]??null};}''',dict(sec=sec,r=row));sections.append(result)
    page.locator('#native-shard-filter').fill('__Q27_NO_SUCH_OPERATOR_20261027__');assert 'no matching source rows'in page.locator('#profile-content').inner_text();page.locator('#native-shard-filter').fill('')
    tests.append(dict(mode=mode,rank_key=row['rank_key'],json_bytes=row['detail']['json_bytes'],gzip_sha256=row['detail']['gzip_sha256'],json_sha256=row['detail']['json_sha256'],sections=sections))
   page.locator('[data-tab="evidence"]').click();page.locator('#vllm-tasktag-membership').wait_for(state='visible');assert '完成语义尚未验收'in page.locator('#vllm-tasktag-membership').inner_text()
   ids=['gen26-domain','gen28-diagnostic','gen33-code','gen36-sg-output','gen39-sampling','gen41-logits','gen44-logits','vllm-tasktag-membership']
   for card in ids:
    assert page.locator('#'+card).count()==1,card
    assert page.locator('#'+card).is_visible(),card
   links=page.evaluate("()=>[...new Set([...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')).filter(h=>h&&!h.startsWith('#')&&!/^[a-z]+:/i.test(h)&&!h.startsWith('/')))]")
   missing=[h for h in links if not(M/h.split('#')[0]).is_file()]
   assert not missing,missing
   contract_tests.append(dict(mode=mode,visible_cards=ids,relative_links=len(links),missing_links=missing))
   result=page.evaluate('''()=>{const ck=(x,m)=>{if(!x)throw Error(m)};const x=DATA.new_measurements.postfreeze_root_audit;ck(!x.status.overall_task_complete,'root pending');ck(x.vllm_tasktag_candidate_membership.summary.accepted_completion_fence_count===0,'no completion promotion');ck(document.getElementById('postfreeze-rank').options.length===4,'SG4');ck(Object.keys(PORTABLE_CPU_SHARDS).length===41,'CPU41');return{prior_CPU_shards:41,prior_SG_ranks:4,candidate_ranks:document.getElementById('vllm-tagged-rank').options.length,outer_semantics_accepted:false}}''')
   page.screenshot(path=str(O/'evidence-desktop.png'));page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(O/'evidence-mobile.png'))
  version=browser.version;browser.close()
 assert not errors and not console,(errors,console)
finally:server.shutdown();server.server_close();thread.join(timeout=5)
out=dict(status='real_single_rank_browser_compatibility_passed_not204'if a.diagnostic_single_rank else'actual204_browser_selection_QA_passed_pending_root_review',observed_ns=time.time_ns(),browser_version=version,html_sha256=sha((M/'index.html').read_bytes()),combined_map_sha256=sha((M/'combined-map.json').read_bytes()),index_identities=expected,actual204_executed=not a.diagnostic_single_rank,selections=tests,retained=result,restored_contracts=contract_tests,page_errors=errors,console_errors=console,local_server_terminated=not thread.is_alive(),overall_complete=False)
(O/'browser-QA.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
