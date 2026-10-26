#!/usr/bin/env python3
"""Real grammar cache/FSM and original MTP host-tree masks; no native/model performance."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import concurrent.futures,datetime,hashlib,json,pathlib,sys,traceback
from types import SimpleNamespace as NS
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L032-R005-grammar-cpu';O.mkdir(parents=True,exist_ok=False);checks=[];cases=[];current='load';F=None;backends=[]
def ck(v,label):
 checks.append({'case':current,'check':label,'pass':bool(v)})
 if not v:raise AssertionError(current+': '+label)
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def newbackend():
 b=F.backend();backends.append(b);return b
def fresh(seed,ids=()):
 g=seed.copy()
 for tid in ids:g.accept_token(int(tid))
 return g
try:
 import L032_grammar_components as F
 import jsonschema,torch
 ck(F.V==154856,'historical tokenizer vocabulary fixture size');ck(F.tokenizer.is_fast,'real historical HF fast tokenizer');b=newbackend();seed=b.dispatch_json(F.KEY[1]);ck(isinstance(seed,F.Grammar),'actual XGrammar backend compiles nested schema');initial=F.snapshot(seed)
 for bs in [2,3]:
  current=f'cache-cold-hot-c{bs}';b0=newbackend();gm=F.manager(b0);reqs=[F.Request('cold'+str(i)) for i in range(bs)]
  for q in reqs:ck(gm.process_req_with_grammar(q),'cold miss queues real Future')
  ck(all(isinstance(q.grammar,concurrent.futures.Future) for q in reqs),'real compilation futures')
  for q in reqs:q.grammar.result(timeout=90)
  ready=gm.get_ready_grammar_requests();ck({q.rid for q in ready}=={q.rid for q in reqs} and not gm.grammar_queue,'manager drains ready set');ck(len({id(q.grammar.matcher) for q in reqs})==bs,'cold requests get independent matchers');cached=b0.cache[F.KEY];cachemask=F.snapshot(cached);ids=F.doc_ids(F.DOCS[0]);reqs[0].grammar.accept_token(ids[0]);ck(torch.equal(F.snapshot(reqs[1].grammar),cachemask),'consuming one request does not mutate peer');ck(torch.equal(F.snapshot(cached),cachemask),'immutable cache seed not consumed')
  hot=[F.Request('hot'+str(i)) for i in range(bs)]
  for q in hot:ck(not gm.process_req_with_grammar(q),'cache hit does not queue');ck(q.grammar.ctx is cached.ctx,'cache hit shares compiled context');ck(q.grammar.matcher is not cached.matcher,'cache hit owns matcher');ck(q.grammar.grammar_stats.is_cache_hit,'source cache-hit stats');ck(torch.equal(F.snapshot(q.grammar),cachemask),'hot matcher starts at initial FSM')
  ck(len({id(q.grammar.matcher) for q in hot})==bs,'hot request matchers isolated');ck(len({id(q.grammar.grammar_stats.tree_traversal_time) for q in hot})==bs,'mutable stats lists isolated');copied=reqs[0].grammar.copy();ck(not copied.accepted_tokens and torch.equal(F.snapshot(copied),cachemask),'copy is fresh matcher, not a checkpoint clone');saved=hot[0].grammar;b0.reset();ck(not b0.cache,'reset clears cache')
  for tid in F.doc_ids(F.DOCS[1]):saved.accept_token(tid)
  saved.accept_token(F.EOS);ck(saved.is_terminated(),'existing request context remains usable after compiler cache reset');cases.append({'id':current,'cold_requests':bs,'hot_requests':bs,'fresh_matcher_copy_semantics':True,'compiler_internal_dedup_measured':False})
 for i,doc in enumerate(F.DOCS):
  current=f'nested-json-accept-EOS-{i}';jsonschema.Draft202012Validator(F.SCHEMA).validate(doc);ids=F.doc_ids(doc);g=fresh(seed)
  for pos,tid in enumerate(ids):
   mask=F.snapshot(g);bits=F.allowed(mask[0]);ck(bool(bits[tid]),'gold document next token allowed');ck(not bool(bits[F.EOS]),'EOS rejected before complete JSON');g.accept_token(tid)
  complete=F.snapshot(g);ck(bool(F.allowed(complete[0])[F.EOS]),'EOS allowed at completed JSON');g.accept_token(F.EOS);ck(g.is_terminated(),'EOS terminates real matcher');before=list(g.accepted_tokens);g.accept_token(0);ck(g.accepted_tokens==before,'source ignores additional accept after termination');g.rollback(1);ck(not g.is_terminated() and torch.equal(F.snapshot(g),complete),'rollback EOS restores completed JSON mask')
  for k in [1,3]:
   q=fresh(seed,ids);q.rollback(k);gold=fresh(seed,ids[:-k]);ck(torch.equal(F.snapshot(q),F.snapshot(gold)),'rollback matches independent replay');ck(q.accepted_tokens==ids[:-k],'accepted history follows rollback')
  cases.append({'id':current,'document_tokens':len(ids),'document_sha256':sha(json.dumps(doc,ensure_ascii=False)),'independent_jsonschema_validated':True})
 bad_docs=[{'kind':'red','n':2,'items':[True]},{'kind':'红','n':12,'items':[True]},{'kind':'blue','n':1,'items':[]},{'kind':'红','n':2,'items':[True],'extra':1}]
 for i,doc in enumerate(bad_docs):
  current=f'invalid-document-{i}'
  try:jsonschema.Draft202012Validator(F.SCHEMA).validate(doc);ck(False,'independent schema must reject invalid fixture')
  except jsonschema.ValidationError:pass
  g=fresh(seed);rejected=False
  for tid in F.doc_ids(doc):
   before=F.snapshot(g);bits=F.allowed(before[0])
   if not bool(bits[tid]):
    history=list(g.accepted_tokens)
    try:g.accept_token(tid);ck(False,'invalid token must raise')
    except ValueError:pass
    ck(g.accepted_tokens==history,'failed accept does not append history');ck(torch.equal(F.snapshot(g),before),'failed accept preserves matcher mask');rejected=True;break
   g.accept_token(tid)
  ck(rejected,'grammar rejects independently-invalid document');cases.append({'id':current,'rejected':True,'document_sha256':sha(json.dumps(doc,ensure_ascii=False)),'current_token_is_diagnostic_not_checkpoint':True})
 current='mask-apply-and-mixed-active-rows'
 active=fresh(seed,F.doc_ids(F.DOCS[0])[:3]);finished=fresh(seed);finished.finished=True;terminated=fresh(seed,F.doc_ids(F.DOCS[1]));terminated.accept_token(F.EOS)
 for grammars in [[active,None],[active,finished,None],[active,terminated,None],[finished,terminated,None]]:
  info=F.Sampling();info.grammars=grammars;info.vocab_size=F.V;info.temperatures=torch.ones(len(grammars));info.device=torch.device('cpu');info.update_regex_vocab_mask();mask=info.grammar_mask.vocab_mask;ck(mask.dtype==torch.int32 and mask.shape==(len(grammars),(F.V+31)//32),'original packed mask dtype/shape')
  expected=[]
  for row,g in enumerate(grammars):
   if g is not None and not g.finished and not g.is_terminated():ck(torch.equal(mask[row],F.snapshot(g)[0]),'active row matches individual fill')
   else:ck(bool(torch.all(mask[row]==-1)),'omitted row remains unconstrained')
   expected.append(F.allowed(mask[row]))
  logits=torch.arange(len(grammars)*F.V,dtype=torch.float32).reshape(len(grammars),F.V);gold=logits.clone().masked_fill_(~torch.stack(expected),float('-inf'));info.grammar_mask.apply(logits);ck(torch.equal(logits,gold),'real CPU mask apply matches independent packed-bit oracle')
 active.finished=True;info.grammars=[active,None];info.temperatures=torch.ones(2);info.update_regex_vocab_mask();ck(bool(torch.all(info.grammar_mask.vocab_mask==-1)),'next batch resets old constrained rows');cases.append({'id':current,'mixed_batches':4,'real_xgrammar_CPU_apply':True,'native_NPU_apply':False})
 current='allNone-precondition-witness'
 info=F.Sampling();info.grammars=[None,None];info.vocab_size=F.V;info.temperatures=torch.ones(2);info.device=torch.device('cpu')
 try:info.update_regex_vocab_mask();ck(False,'allNone list precondition witness should raise')
 except StopIteration:cases.append({'id':current,'expected_exception':'StopIteration','caller_invariant_not_proven_in_complete_runtime':True,'current_service_bug_claim':False})
 current='tokenfilter-packed-boundaries'
 tokens=[0,31,32,F.EOS,F.V-1,31]
 for allow in [True,False]:
  for reset in [True,False]:
   m=torch.full((3,(F.V+31)//32),-1 if reset else 0,dtype=torch.int32);before=m.clone();F.Backend.set_token_filter(m,tokens,1,is_allowed=allow,reset_vocab_mask=reset);ck(torch.equal(m[0],before[0]) and torch.equal(m[2],before[2]),'token filter does not touch other rows');gold=F.allowed(before[1])
   if reset:gold.fill_(not allow)
   gold[tokens]=allow;ck(torch.equal(F.allowed(m[1]),gold),'duplicate/sign/word-boundary filter matches set oracle')
 cases.append({'id':current,'option_pairs':4,'tokens_fixture':tokens})
 for bs in [2,3]:
  current=f'MTP-host-chain-c{bs}-draft4';ids_list=[F.doc_ids(F.DOCS[i%3]) for i in range(bs)];prefixes=[ids[:2+i%2] for i,ids in enumerate(ids_list)];grammars=[fresh(seed,p[:-1]) for p in prefixes];draft=torch.tensor([[p[-1],*ids[len(p):len(p)+3]] for p,ids in zip(prefixes,ids_list)],dtype=torch.int64);children=torch.tensor([[1,2,3,-1]]*bs);siblings=torch.full((bs,4),-1,dtype=torch.int64);tree=F.Tree.from_host(children,siblings,draft);events=[]
  class ObservedTree:
   def resolve(self):events.append('resolve');return tree.resolve()
  def barrier():
   events.append('barrier')
   for g,p in zip(grammars,prefixes):g.accept_token(p[-1])
  info=NS(vocab_size=F.V,grammar_mask='stale-extend-mask');out=F.ns['build_grammar_vocab_mask'](reqs=[NS(grammar=g) for g in grammars],tree=ObservedTree(),sampling_info=info,device='cpu',barrier=barrier);ck(events==['barrier','resolve'],'barrier precedes tree resolution');ck(info.grammar_mask is None,'stale extend grammar mask cleared');ck(out.vocab_mask.shape==(bs*4,(F.V+31)//32),'MTP row shape matches flattened verify tree')
  for row,(g,p,ids) in enumerate(zip(grammars,prefixes,ids_list)):
   for depth in range(4):gold=fresh(seed,ids[:len(p)+depth]);ck(torch.equal(out.vocab_mask[row*4+depth],F.snapshot(gold)[0]),'tree row matches independent sequential replay')
   ck(torch.equal(F.snapshot(g),F.snapshot(fresh(seed,p))),'DFS restores original FSM mask');ck(g.accepted_tokens==p,'DFS restores committed history')
  logits=torch.zeros(bs*4,F.V);out.apply(logits);expect=torch.stack([F.allowed(r) for r in out.vocab_mask]);ck(torch.equal(torch.isfinite(logits),expect),'MTP built mask actually applies to CPU logits');cases.append({'id':current,'draft_rows':4,'scope':'Historical3/1/4-style synthetic host chain, not model acceptance or device execution','events':events})
 current='MTP-rejected-siblings-and-padded-ID'
 ids=F.doc_ids(F.DOCS[0]);prefix=ids[:2];g=fresh(seed,prefix);ck(not bool(F.allowed(F.snapshot(g)[0])[F.EOS]),'EOS is invalid sibling at chosen prefix');draft=torch.tensor([[prefix[-1],ids[2],F.EOS,F.V+4]]);children=torch.tensor([[1,-1,-1,-1]]);siblings=torch.tensor([[-1,2,3,-1]]);mask,handle=F.ns['generate_token_bitmask']([NS(grammar=g)],children,siblings,draft,F.V);ck(torch.equal(mask[0],F.snapshot(fresh(seed,prefix))[0]),'root mask');ck(torch.equal(mask[1],F.snapshot(fresh(seed,ids[:3]))[0]),'accepted sibling row');ck(bool(torch.all(mask[2:]==-1)),'rejected/out-of-range rows unvisited and left unconstrained');ck(torch.equal(F.snapshot(g),F.snapshot(fresh(seed,prefix))) and g.accepted_tokens==prefix,'sibling traversal restores request state');cases.append({'id':current,'native_acceptance_not_proven':True,'unvisited_rows_need_parent_mask_acceptance_in_runtime':True})
 current='invalid-schema-and-cache'
 unsupported={'type':'array','items':{'type':'integer'},'uniqueItems':True};badkey=('json',json.dumps(unsupported));gm=F.manager(b);q=F.Request('bad-schema',badkey);ck(gm.process_req_with_grammar(q),'invalid schema initially queues');q.grammar.result(timeout=60);gm.get_ready_grammar_requests();ck(q.finished() and isinstance(q.grammar,F.Invalid),'unsupported semantic constraint not silently weakened');hot=F.Request('bad-schema-hot',badkey);ck(not gm.process_req_with_grammar(hot) and hot.finished(),'invalid cache aborts next request')
 for key in [('regex','a\x00b'),('json',json.dumps({'type':'string','pattern':'a\x00b'}))]:value=b._init_value_dispatch(key,False);ck(isinstance(value,F.Invalid),'NUL rejected before native grammar converter')
 cases.append({'id':current,'invalid_cache_policy':'persistent until backend reset','unsupported_constraint':'uniqueItems','NUL_cases':2})
 current='queue-abort-pending-running-futures'
 bq=newbackend();gm=F.manager(bq);f0=concurrent.futures.Future();f1=concurrent.futures.Future();f1.set_running_or_notify_cancel();q0=F.Request('abort-a');q1=F.Request('abort-b');q0.grammar=f0;q1.grammar=f1;gm.grammar_queue=[q0,q1];gm.abort_requests(NS(rid='abort-',abort_all=False));ck(q0.finished() and q1.finished() and f0.cancelled() and not f1.cancelled(),'pending cancel vs running future ownership');returned=gm.get_ready_grammar_requests();ck(len(returned)==2 and not gm.grammar_queue and not bq.cache,'aborted queue drained without publishing cache');f1.set_result(seed.copy());ck(not bq.cache,'late running completion does not publish ghost cache');cases.append({'id':current,'Future_job_boundary':'controlled Future, not delayed actual compiler or scheduler','native_reclaim':False})
 for running in [False,True]:
  current=f'queue-timeout-cache-policy-running{running}';bt=newbackend();gm=F.manager(bt);gm.SGLANG_GRAMMAR_POLL_INTERVAL=.001;gm.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS=1;f=concurrent.futures.Future()
  if running:f.set_running_or_notify_cancel()
  q=F.Request('timeout');q.grammar=f;q.grammar_key=F.KEY;gm.grammar_queue=[q];returned=gm.get_ready_grammar_requests();ck(returned==[q] and q.finished(),'controlled timeout aborts request');ck(isinstance(bt.cache[F.KEY],F.Invalid),'timeout caches InvalidGrammarObject')
  if running:f.set_result(seed.copy())
  nextq=F.Request('after-timeout');ck(not gm.process_req_with_grammar(nextq) and nextq.finished(),'next same-key request sees cached timeout');bt.reset();ck(not bt.cache,'reset clears cached timeout');cases.append({'id':current,'policy_not_current_service_failure':True,'actual_compile_timing_not_tested':True})
 result={'status':'PASS','semantic_checks':len(checks),'case_count':len(cases),'cases':cases,'checks':checks,'extracted':F.records,'boundaries':F.BOUNDARIES,'local_environment':{'torch':torch.__version__,'xgrammar':'0.2.7','vocab_fixture':F.V,'EOS_fixture':F.EOS},'real_grammar_compiler_and_matcher_executed':True,'real_CPU_apply_executed':True,'source_MTP_host_DFS_executed':True,'source_single_rank_grammar_queue_executed':True,'native_executed':False,'whole_serving_modules_imported':False,'model_config_identity_confirmed':False,'performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','case_count':len(cases),'semantic_checks':len(checks)}))
except BaseException as e:
 (O/'result.json').write_text(json.dumps({'status':'FAIL','failed_case':current,'error_type':type(e).__name__,'error':str(e),'completed_cases':cases,'checks':checks,'native_executed':False,'performance':'NOT_MEASURED'},ensure_ascii=False,indent=2)+'\n');traceback.print_exc();sys.exit(1)
finally:
 for b in backends:b.executor.shutdown(wait=True,cancel_futures=True)
