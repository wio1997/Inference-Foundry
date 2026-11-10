// Local artifact UI smoke check. Chrome only opens our local HTML, no public site.
import {spawn} from 'node:child_process';
import {readFile,mkdir,writeFile,mkdtemp} from 'node:fs/promises';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {dirname,resolve} from 'node:path';
import {tmpdir} from 'node:os';
const folder=dirname(fileURLToPath(import.meta.url));
const profile=await mkdtemp(resolve(tmpdir(),'wio171-global-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
let ws,seq=0;const pending=new Map(),errors=[];
try{
let port;for(let i=0;i<100;i++){try{port=Number((await readFile(resolve(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);break;}catch{await delay(100);}}
if(!port)throw Error('No local Chrome debugging port');
const pages=await fetch('http://127.0.0.1:'+port+'/json/list').then(r=>r.json());
ws=new WebSocket(pages[0].webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const assert=(value,message)=>{if(!value)throw Error(message);};
await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url:pathToFileURL(resolve(folder,'../../171-框架Profile热点.html')).href});
for(let i=0;i<100;i++){if(await evaluate("document.querySelector('#op-rows')?.children.length>0"))break;await delay(100);}
const initial=await evaluate("({nodes:document.querySelectorAll('[data-node]').length,ops:document.querySelector('#op-rows').children.length,ranks:document.querySelector('#rank-table').children.length,requests:document.querySelector('#request-rows').children.length,error:document.querySelector('#error').textContent,title:document.querySelector('#detail h2').textContent})");
assert(initial.nodes===18&&initial.ops===84&&initial.ranks===16&&initial.requests===8&&!initial.error,'Initial population: '+JSON.stringify(initial));
assert(initial.title.includes('两组'),'Stale DP1-only idle label');
assert(await evaluate("!document.querySelector('#bill-panel').hidden && document.querySelector('#overview-panel').hidden && document.querySelectorAll('.bill-op').length===6"),'Plain summary is not the default');
await evaluate("document.querySelector('[data-view=overview]').click()");
await evaluate("document.querySelector('[data-node=moe]').click()");await delay(50);
assert(await evaluate("!!document.querySelector('#detail .operator-hint')"),'Missing module-to-kernel link');
await evaluate("document.querySelector('#detail .operator-hint').click()");
assert(await evaluate("!document.querySelector('#kernels-panel').hidden && [...document.querySelectorAll('#op-rows tr')].every(r=>r.textContent.includes('MoE'))"),'Module drilldown');
await evaluate("document.querySelector('#op-family').value='all';document.querySelector('#op-family').dispatchEvent(new Event('change'));document.querySelector('#op-dp').value='dp1';document.querySelector('#op-dp').dispatchEvent(new Event('change'))");
assert(await evaluate("document.querySelector('#op-detail').textContent.includes('162.74')"),'DP1 cost mismatch');
await evaluate("document.querySelector('#op-search').value='engram';document.querySelector('#op-search').dispatchEvent(new Event('input'))");
assert(await evaluate("document.querySelectorAll('#op-rows tr').length>0 && [...document.querySelectorAll('#op-rows tr')].every(r=>r.textContent.toLowerCase().includes('engram'))"),'Search');
await evaluate("document.querySelector('#op-search').value='';document.querySelector('#op-search').dispatchEvent(new Event('input'))");
const shot=async name=>{const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(resolve('/tmp',name),Buffer.from(r.data,'base64'));};
await shot('171-global-kernels.png');
await evaluate("document.querySelector('[data-view=ranks]').click()");
assert(await evaluate("!document.querySelector('#ranks-panel').hidden && document.querySelector('#timeline-caption').textContent.includes('每格')"),'Timeline view');
await shot('171-global-ranks.png');
await evaluate("document.querySelector('#rank-phase').value='early';document.querySelector('#rank-phase').dispatchEvent(new Event('change'))");
assert(await evaluate("Number(document.querySelector('#rank-table tr').children[5].textContent.split('/')[0])>500"),'Early table update');
await evaluate("document.querySelector('[data-view=calls]').click()");
assert(await evaluate("document.querySelector('#call-detail').textContent.includes('24.15') && document.querySelector('#call-detail').textContent.includes('52532')"),'Same-call linkage');
await shot('171-global-calls.png');
await evaluate("document.querySelector('#call-rank').value='8';document.querySelector('#call-rank').dispatchEvent(new Event('change'))");
assert(await evaluate("document.querySelector('#call-detail').textContent.includes('EP8')"),'Per-rank call selector');
await evaluate("document.querySelector('[data-view=scope]').click();document.querySelector('#request-round').value='profiled';document.querySelector('#request-round').dispatchEvent(new Event('change'))");
assert(await evaluate("document.querySelector('#request-rows').children.length===8"),'Request rows');
await evaluate("document.querySelector('[data-view=overview]').click();document.querySelector('[data-node=idle]').click();window.scrollTo(0,0)");await shot('171-global-chain.png');
await evaluate("document.querySelector('[data-view=bill]').click();window.scrollTo(0,0)");await shot('171-plain-time-bill.png');
await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
const mobile=await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth})");assert(mobile.scroll<=mobile.width,'Mobile overflow '+JSON.stringify(mobile));
assert(errors.length===0,'Browser errors '+JSON.stringify(errors));
const result={passed:true,local_only:true,initial,mobile,checks:['plain time bill default','module drilldown','operator search','DP switch','16-rank timeline','early/steady','same-call flow','per-rank calls','8 request timings','mobile layout'],screenshots:['/tmp/171-plain-time-bill.png','/tmp/171-global-chain.png','/tmp/171-global-ranks.png','/tmp/171-global-kernels.png','/tmp/171-global-calls.png']};
await writeFile(resolve(folder,'view-validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{if(ws)ws.close();chrome.kill();}
