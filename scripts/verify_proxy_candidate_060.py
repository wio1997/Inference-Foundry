#!/usr/bin/env python3
"""Check local proxy candidate hashes and selected source-level AST relations.

This proves neither the formal runtime file nor individual request execution.
Local paths/source contents/addresses are not emitted.
"""
import argparse,ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/p_latency_scope_060/proxy_candidate_v2.json'
SHA={'proxy':'85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469','manifest':'2f627d8df32a3f4b5ebd2ea36602998d3c24e886aa3fec62280fbed0624b3c3a','launcher':'abd479f506cc3aa121fa5a08fa085a97e82c6532b2ce5e7badd154d707a346d1','stats':'af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3','loggers':'c12b4fbebfe886b697127f1114eaa0bb83c54a78675919d178b619c3d8d5d655'}


def need(x,message):
 if not x:raise ValueError(message)


def read_hash(path,expected):
 p=Path(path);need(p.is_file() and not p.is_symlink(),'input not regular');st=p.stat();need(st.st_size<5000000,'input too large');raw=p.read_bytes();after=p.stat();need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'changed input');need(hashlib.sha256(raw).hexdigest()==expected,'candidate SHA mismatch');return raw


def find_fn(tree,name):
 xs=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name];need(len(xs)==1,'candidate function mismatch');return xs[0]


def call_name(node):
 if not isinstance(node,ast.Call):return None
 if isinstance(node.func,ast.Name):return node.func.id
 if isinstance(node.func,ast.Attribute):return node.func.attr
 return None


def facts(source):
 tree=ast.parse(source);build=find_fn(tree,'build_prefill_request');assign=find_fn(tree,'assign_instances');handle=find_fn(tree,'handle_completions_impl');out={}
 for key,value in (('stream',False),('max_tokens',1),('min_tokens',1)):
  hits=[x for x in ast.walk(build) if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Subscript) and isinstance(x.targets[0].value,ast.Name) and x.targets[0].value.id=='payload' and isinstance(x.targets[0].slice,ast.Constant) and x.targets[0].slice.value==key and isinstance(x.value,ast.Constant) and type(x.value.value)==type(value) and x.value.value==value];need(len(hits)==1,'candidate prefill configuration');out['prefill_'+key]=value
 pending=[x for x in ast.walk(assign) if isinstance(x,ast.Await) and call_name(x.value)=='send_request_to_service'];picked=[x for x in ast.walk(assign) if isinstance(x,ast.Await) and call_name(x.value)=='schedule' and x.value.args and isinstance(x.value.args[0],ast.Constant) and x.value.args[0].value=='pick_decoder'];need(len(pending)==len(picked)==1 and pending[0].lineno<picked[0].lineno,'candidate assignment ordering')
 handoff=[x for x in ast.walk(handle) if isinstance(x,ast.Await) and call_name(x.value)=='assign_instances'];opening=[x for x in ast.walk(handle) if isinstance(x,ast.Call) and call_name(x)=='stream_service_response'];need(len(handoff)==1 and opening and handoff[0].lineno<min(x.lineno for x in opening),'candidate stream ordering')
 first=[x for x in ast.walk(handle) if isinstance(x,ast.Await) and call_name(x.value)=='__anext__'];need(first and min(x.lineno for x in opening)<min(x.lineno for x in first),'candidate first-chunk ordering')
 out.update({'prefill_post_await_precedes_decode_selection_in_source':True,'instance_assignment_await_precedes_decode_stream_open_in_source':True,'decode_stream_open_precedes_first_chunk_await_in_source':True});return out


def count_value(x,needle):
 if isinstance(x,dict):return sum(count_value(v,needle) for v in x.values())
 if isinstance(x,list):return sum(count_value(v,needle) for v in x)
 return int(x==needle)


def summarize(a):
 blobs={key:read_hash(getattr(a,key),expected) for key,expected in SHA.items()};need(count_value(json.loads(blobs['manifest']),SHA['proxy'])==1,'proxy SHA absent from candidate manifest')
 return {'schema':'glm53-vllm-proxy-candidate-static-v1','candidate_file_sha256':SHA,'candidate_manifest_records_proxy_sha':True,'static_ast':facts(blobs['proxy'].decode('utf8')),'limitations':['Candidate source ordering is static only; no formal G47/G48 proxy process or runtime source/import SHA has been verified','P response completion is not proven KV physical network transfer completion','Engine metrics first-token observation is not necessarily HTTP chunk delivery or client first frame','P and D server counters may contain different requests and overlapping lifecycle intervals']}


def main():
 ap=argparse.ArgumentParser()
 for key in SHA:ap.add_argument('--'+key,required=True)
 ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args();d=summarize(a);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,SyntaxError,UnicodeError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: candidate integrity or AST validation failed')
