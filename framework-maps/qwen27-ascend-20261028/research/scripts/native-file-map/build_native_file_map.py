"""Add lazy local-script native sidecars to a fresh copy of a completed map.

The original gzip bytes and their gzip/raw SHA and identity checks are retained.
This only changes file: transport. HTTP retains the original fetch path.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,shutil,time

OLD="""  else{
   const base=source.detail_base_href;if(typeof base!=='string')throw new Error('detail_base_href missing and no inline shard exists');"""
NEW="""  else if(location.protocol==='file:'){
   const base=source.detail_base_href;
   if(typeof base!=='string'||!safeRelativePath(base))throw new Error('unsafe local native base');
   const href=new URL(rel+'.js',new URL(base.endsWith('/')?base:base+'/',document.baseURI));
   const encoded=await new Promise((resolve,reject)=>{
    const script=document.createElement('script');let finished=false,timer;
    const cleanup=()=>{clearTimeout(timer);script.remove();signal.removeEventListener('abort',abort)};
    const done=(error)=>{if(finished)return;finished=true;cleanup();
     const data=window.NATIVE_DETAIL_SHARDS?.[rel];if(window.NATIVE_DETAIL_SHARDS)delete window.NATIVE_DETAIL_SHARDS[rel];
     if(error)reject(error);else if(typeof data!=='string')reject(new Error('local native sidecar payload missing'));else resolve(data)};
    const abort=()=>done(new DOMException('Native selection cancelled','AbortError'));
    script.onload=()=>done();script.onerror=()=>done(new Error('local native sidecar could not be loaded'));
    timer=setTimeout(()=>done(new Error('local native sidecar timeout')),60000);
    signal.addEventListener('abort',abort,{once:true});if(signal.aborted){abort();return}
    script.src=href.href;document.head.append(script);
   });
   if(signal.aborted)throw new DOMException('Native selection cancelled','AbortError');
   compressed=bytesFromBase64(encoded);
  }
  else{
   const base=source.detail_base_href;if(typeof base!=='string')throw new Error('detail_base_href missing and no inline shard exists');"""

def sha(b):return hashlib.sha256(b).hexdigest()
def build(source,output,diagnostic=False):
 source=Path(source).resolve();output=Path(output).resolve()
 if output.exists():raise ValueError('output must be fresh')
 if source==output or source in output.parents:raise ValueError('output cannot be nested in source')
 combined=source/'combined-map.json';data=json.loads(combined.read_text());ix=data['new_measurements']['native_sharded_map']['index'];expected=1 if diagnostic else 204
 assert ix['identity_count']==len(ix['per_rank'])==expected and not ix['bundle_errors']
 assert len({r['rank_key']for r in ix['per_rank']})==expected
 if not diagnostic:
  c=json.loads((source/'composition-receipt.json').read_text());assert c['verified_identity_count']==204 and c['output_combined_map']['sha256']==sha(combined.read_bytes())
 html=(source/'index.html').read_text();assert html.count(OLD)==1 and NEW not in html
 entries=list(ix['per_rank']);all_sidecars=[];files=[]
 for p in source.rglob('*'):
  if p.is_symlink():raise ValueError('source symlink refused')
  if p.is_file():files.append(p)
 output.mkdir(parents=True)
 for p in files:
  q=output/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
  if p.name=='index.html':q.write_text(html.replace(OLD,NEW,1));continue
  try:os.link(p,q)
  except OSError:shutil.copyfile(p,q)
 # Original checked rank payloads are encoded verbatim; browser rechecks both hashes.
 for row in entries:
  r=row['detail'];rel=Path(r['path']);assert rel.name==r['path']and not rel.is_absolute()
  p=source/'native-rank-details'/rel;raw=p.read_bytes();assert len(raw)==r['gzip_bytes'] and sha(raw)==r['gzip_sha256']
  out=output/'native-rank-details'/(rel.name+'.js')
  js=('window.NATIVE_DETAIL_SHARDS=window.NATIVE_DETAIL_SHARDS||Object.create(null);window.NATIVE_DETAIL_SHARDS['+json.dumps(rel.name)+']='+json.dumps(base64.b64encode(raw).decode())+';\n').encode();out.write_bytes(js)
  all_sidecars.append(dict(path=str(out.relative_to(output)),bytes=len(js),sha256=sha(js),source_gzip_sha256=r['gzip_sha256'],source_json_sha256=r['json_sha256'],rank_key=row['rank_key']))
 result=dict(status='diagnostic1_file_transport_built_not204'if diagnostic else'actual204_file_native_transport_built_pending_browser_QA',created_ns=time.time_ns(),source=str(source),output=str(output),source_html_sha256=sha((source/'index.html').read_bytes()),html_sha256=sha((output/'index.html').read_bytes()),combined_map_sha256=sha(combined.read_bytes()),source_files_preserved=len(files)-1,native_sidecar_count=len(all_sidecars),sidecars=all_sidecars,transport_scope='file: selected native gzip through a local script; HTTP fetch unchanged; compressed/raw SHA and exact identity checks unchanged; selected payload discarded after decode',actual204_executed=not diagnostic,overall_complete=False)
 (output/'native-file-build-receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic-single-rank',action='store_true');a=p.parse_args();r=build(a.source,a.output,a.diagnostic_single_rank);print(json.dumps({k:v for k,v in r.items()if k!='sidecars'}))
