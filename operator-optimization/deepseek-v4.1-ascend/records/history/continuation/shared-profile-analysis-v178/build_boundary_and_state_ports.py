"""Conservative public contracts, not a claim about a native schedule or dead values."""
import collections
import hashlib
import json
from pathlib import Path

N = Path('/Users/wio/work/166-vllm-analysis-continuation-20261030')
O = N/'profile-first-20261031'
H = N/'shared-profile-analysis-v178'
def read(p): return json.loads(p.read_text())
def pin(p): return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(n,x): (H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

# Input/output names below are semantic interfaces; different named versions may alias physically.
specs = [
 ('HOST_TOKENIZE', ['M01'],
  ['request','validated_request','tokens','resolved_sampling_contract'],
  [('request','validated_request','original endpoint/schema/text-only guards'),
   ('validated_request','tokens','tokenizer identity, special-token and truncation semantics'),
   ('validated_request','resolved_sampling_contract','original defaults/overrides, RNG and output requirements')],
  'H04 may batch/cache identical tokenizer work only with tokenizer/options/input identity; preserve errors and request isolation.'),
 ('HOST_PREFIX_HASH', ['M02','M24'],
  ['tokens','namespace','hash_chain','prefix_lookup','page_leases','retirement'],
  [('tokens','hash_chain','ordered prefix and original block boundaries'),('namespace','hash_chain','salt/model/cache/hash namespace'),
   ('hash_chain','prefix_lookup','exact equality verification and eligible prefix'),('prefix_lookup','page_leases','ownership/refcounts/generation'),
   ('retirement','page_leases','recycle only after all readers/writers retire')],
  'H03 may reuse prefix/hash work under the same namespace and immutable token prefix; hit counters do not attest physical page identity.'),
 ('HOST_INPUT_PACK', ['M03','M04','M21','M24'],
  ['schedule','request_generation','previous_sampled_ids','prefix_reused_rows','row_map','packed_input','positions','counts','slots'],
  [('schedule','row_map','exact scheduled/new/speculative row membership and ordering'),('request_generation','row_map','request identity and generation'),
   ('previous_sampled_ids','packed_input','direct-prefix/permuted fallback of accepted previous results'),
   ('prefix_reused_rows','packed_input','original reuse/untouched row policy'),('row_map','packed_input','gather/scatter map'),
   ('row_map','positions','logical query position'),('row_map','counts','actual versus padded rows'),('row_map','slots','per-request page/slot mapping')],
  'H02 combines preparation and publication per identical row generation; capacity, logical count and processed work remain distinct.'),
 ('METADATA_AND_STATE_PUBLICATION', ['M04','M08','M09','M10','M18','M21','M22','M24'],
  ['row_map','counts','positions','page_leases','slots','accepted_lengths_old','state_generation','attention_metadata','index_metadata','draft_metadata','state_publication'],
  [('row_map','attention_metadata','request segmentation and original padding'),('counts','attention_metadata','valid/padded lengths'),
   ('positions','attention_metadata','positions and sequence lengths'),('slots','attention_metadata','exact physical slots'),
   ('page_leases','attention_metadata','block table generation'),('attention_metadata','index_metadata','selected compression/index mapping'),
   ('attention_metadata','draft_metadata','original draft expansion and query/context row maps'),
   ('accepted_lengths_old','state_publication','committed prefix from previous iteration, never future acceptance'),
   ('state_generation','state_publication','owner/request/allocation version'),('draft_metadata','state_publication','exact writes to graph-stable state')],
  'F31 fuses arithmetic/copy/publication only with all consumers and bank epochs retained. BOOL[24] is a view, not an allocation identity.'),
 ('GRAPH_INPUT_BANK', ['M00','M04','M21','M22','M24'],
  ['host_staging','bank_old','metadata','last_readers','bank_write','capture_or_update','replay_or_eager','bank_new'],
  [('host_staging','bank_write','original async copy/packing completion'),('bank_old','bank_write','untouched live elements'),
   ('metadata','bank_write','valid ranges and generation'),('last_readers','bank_write','overlapping bank reuse only after retirement'),
   ('bank_write','capture_or_update','graph-safe addresses, shapes and parameter lifetime'),
   ('capture_or_update','replay_or_eager','selected graph/eager/zero-work branch; preserve event and update protocol'),
   ('bank_write','bank_new','same physical bank may contain a different epoch')],
  'H02/H06/H07 may remove repeated packing/copies or alter bank protocol conditionally. Stable pointers do not prove values stable or writes unnecessary.'),
 ('PACKING_BUCKET_DOMAIN', ['M00','M04','M21','M22','M23'],
  ['actual_rows','padded_bucket','validity_mask','collective_participants','selected_call'],
  [('actual_rows','padded_bucket','selected eager/capture bucket'),('actual_rows','validity_mask','valid and dummy rows'),
   ('padded_bucket','selected_call','native selected call geometry'),('validity_mask','selected_call','original dummy/padding semantics'),
   ('collective_participants','selected_call','remote receiver work can remain when local sender rows are zero')],
  'H06 changes bucket/packing only with capture validity and all rank roles; allocated tiles are not active MAC rows.'),
 ('EMBEDDING_ALTERNATIVE', ['M04','M05','M06','M19','M22','M23'],
  ['input_ids','vocab_layout','embedding','shard_or_restore','hc_repeat','output_hidden'],
  [('input_ids','embedding','original invalid/masked IDs and int width'),('vocab_layout','embedding','global/local rows and tied/unshared weights'),
   ('embedding','shard_or_restore','selected SP order'),('shard_or_restore','hc_repeat','target shards before HC repeat; draft order distinct'),
   ('hc_repeat','output_hidden','same rounded BF16 values and alias semantics')],
  'H07/F28 may use IDs, packed embedding output or repeated-row reuse only under exact row/weight identity and consumers. Inputs_embeds is a distinct guarded public route.'),
 ('SCORE_SELECTION_AND_REJECTION', ['M01','M15','M16','M17','M19','M20','M21','M24','M25'],
  ['target_hidden','target_head','target_scores','score_processors','raw_score_observers','target_argmax_or_random','bonus','draft_ids','draft_probabilities','sampling_metadata','rng_old','rng_new','verify_recover','accepted_ids','accepted_lengths','state_commit','output_payload'],
  [('target_hidden','target_head','original collapse/norm/sample-selected rows'),('target_head','target_scores','BF16/logit dtype and TP owner layout'),
   ('target_scores','score_processors','penalties/constraints/temperature selected contract'),
   ('target_scores','raw_score_observers','original optional raw logits/logprobs/entropy side consumers; absent request is not a universal no-reader proof'),
   ('score_processors','target_argmax_or_random','all-greedy/mixed/random guard and exact ties'),
   ('sampling_metadata','target_argmax_or_random','original row/rank/temperature contract'),
   ('target_argmax_or_random','bonus','original bonus row and ID width'),
   ('draft_ids','verify_recover','ordered proposals'),('draft_probabilities','verify_recover','conditional proposal probabilities/default seeded path'),
   ('target_argmax_or_random','verify_recover','actual target sample/score demand'),('bonus','verify_recover','bonus append semantics'),
   ('rng_old','target_argmax_or_random','conditional target random draws under original generator mapping'),
   ('target_argmax_or_random','rng_new','conditional target sampler generator advance'),
   ('rng_old','verify_recover','all original draws/state effects, including fallback draws'),('verify_recover','rng_new','state advance preserved even if values unused'),
   ('sampling_metadata','verify_recover','actual n=1/general, entropy/random/recovery branch'),('verify_recover','accepted_ids','ordered accepted/recovered/bonus IDs'),
   ('verify_recover','accepted_lengths','committed prefix lengths'),('accepted_lengths','state_commit','rollback/free only after actual acceptance'),
   ('accepted_ids','output_payload','public token dtype/shape and request row mapping'),('raw_score_observers','output_payload','conditional requested scores')],
  'F35 can preserve-score argmax then INT32->INT64 public conversion; stronger score/head elimination stays conditional on all readers and exact target/draft weights. No future keep/accept prunes its own producer.'),
 ('HOST_COMPLETION_NOTIFY', ['M03','M22','M24','M25'],
  ['device_completion','completion_payload','notify','consumer_queue'],
  [('device_completion','completion_payload','original event/data readiness'),('completion_payload','notify','same generation and exception/result semantics'),
   ('notify','consumer_queue','request-isolated wakeup, cancellation and lifetime')],
  'H01 can coalesce completion notifications only when every consumer sees the original ready result/generation and no wakeup is lost.'),
 ('HOST_RESPONSE_READY', ['M01','M03','M17','M24','M25'],
  ['accepted_device_results','d2h_ready','request_generation','host_result','detokenized','stop_and_stream','response','release'],
  [('accepted_device_results','d2h_ready','copy after device completion with original layout/dtype'),
   ('d2h_ready','host_result','copy readiness before host inspection'),('request_generation','host_result','stale/cancelled request filtering'),
   ('host_result','detokenized','original token order/tokenizer state'),('detokenized','stop_and_stream','EOS/length/ignore_eos/stream contract'),
   ('stop_and_stream','response','ordered response and optional score payload'),('response','release','release only when queue/buffer readers retire')],
  'H05 can batch or overlap transfer/output preparation only across independent request generations with identical streaming/termination semantics.')
]
templates=[]
for tid,mods,ns,es,answer in specs:
    templates.append({'id':tid,'modules':mods,'nodes':[{'id':x,'kind':'PUBLIC_VALUE_CONTROL_OR_STATE_PORT'} for x in ns],
        'edges':[{'producer':a,'consumer':b,'value':v,'dependency_kind':'PRESERVE_PUBLIC_CONTRACT'} for a,b,v in es],
        'operator_answer':answer,'source_authority':str(O/'module-phase-branch-matrix-v177.json'),
        'boundary_limit':'Conservative source-contract region. Exact alias ranges, native parameter internals and request critical path are separate unknowns; no deletion or independence is licensed.'})

states = [
 ('REQUEST_INPUT_BANKS',['M04','M05','M14','M18','M21','M22','M24'],
  ['previous accepted IDs','host packing','device copy/scatter'],['target/draft embedding','Engram hash','sampling/penalty input','next iteration packing'],
  'request generation, current/previous row map, capture-bank address and valid range',
  'direct-prefix/permuted and untouched-row branches differ; preserve last-reader retirement before mutation'),
 ('POSITION_COUNT_SLOT_METADATA',['M02','M04','M08','M09','M10','M14','M18','M21','M22','M24'],
  ['scheduler/metadata publication'],['SWA and compressed KV','indexer','compressor ring','Engram hash/cache','draft context/query','graph replay'],
  'request,page lease generation,token position,actual/padded rows,rank ownership',
  'share arithmetic only for identical logical domains; identical shape or zero local count is insufficient'),
 ('KV_RING_INDEX_CACHE',['M02','M08','M09','M10','M18','M21','M24'],
  ['SWA scatter','compressed pool/key/scale writes','draft shared-KV scatter','accept/reset/recycle'],
  ['current and future attention','index scoring','C2 old-ring pool','prefix reuse'],
  'cache family,source layer,request,page/slot generation,logical position,committed prefix',
  'write/read/retire/overwrite ordered per physical range; duplicate logical indices retain attention multiplicity'),
 ('TOPK_SHARED_BANK',['M09','M10','M18','M21','M22','M24'],
  ['source layers2,8,14,20,24,28,32,36','conditional draft writer'],['38 target attention layers','conditional draft/native consumers'],
  'worker/device/StorageImpl/allocation generation/query-row range/value epoch',
  'target/draft EP0 shared16MiB backing is positive; preserve 8 source epochs and range WAR/WAW, no cross-rank alias inference'),
 ('C2_BOOL_MASK_BANK',['M08','M09','M10','M21','M22','M24'],
  ['original fills/copies/padding publication'],['selected C2 query masks','graph/native metadata readers'],
  '8192B backing plus view offset/length/generation; observed lengths24,1704,3200,8176',
  'a24-element view is not a24-byte allocation; native MemSet ABI and values are unknown; public replacement retains all original writes'),
 ('ENGRAM_HISTORY_AND_LOOKUP',['M14','M21','M23','M24'],
  ['hash slot-cache updates','INT8 code+FP32 scale lookup/dequant','mask publication'],['hash history','layer1/14 projection/gate','conditional DP/TP consumers'],
  'request/slot/position/engram-layer/table version and original masks',
  'lookup uses positive host-UVA NPU path; overlap flagFalse. Preserve rotated/unrotated and pre-rotation residual paths; table capacity is not traffic'),
 ('MHC_COEFFICIENT_RESIDUAL_AUX',['M06','M14','M15','M18','M19','M24'],
  ['attn/ffn pre and post','optional Engram residual modification'],['same-layer post/comb','next-layer pre_mix','saved target37/38/39 aux','terminal collapse','draft context'],
  'layer,pre_mix generation,rounded residual value and original HC reduction domain',
  'coefficient fork, residual, auxiliary means and terminal pre_mix are distinct uses; Engram breaks a direct residual shortcut'),
 ('DRAFT_CONTEXT_SHARED_KV',['M15','M18','M19','M21','M24'],
  ['ordered target aux concat/proj/norm','context KV projection/RoPE/scatter'],['all3 draft query decoder blocks','future eligible draft steps'],
  'context positions,slots,accepted/request generation,shared cache layer',
  'context normalized hidden is not query embedding and not raw confidence hidden; keep scatter/publication and valid ranges'),
 ('TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS',['M05','M16','M19','M20','M23'],
  ['load/quant/rotation selection'],['target H5120 TP8 head','draft H5120 TP8 head','replicated H256 Markov bias','selected Markov embedding rows'],
  'root,parameter identity,layout,dtype,quant scale/rounding and owner',
  'four target/draft embedding/head storages distinct; this does not prove numerical inequality. Replicated Markov head not TP8. No blanket weight CSE'),
 ('SCORES_SAMPLING_RNG',['M01','M16','M17','M20','M24','M25'],
  ['target/draft head','score processing','Markov steps','RNG draws'],['greedy/random selection','rejection/recovery/bonus','raw logprob/entropy readers','next RNG state'],
  'request,row/head domain,step0..4,score precision,tie policy,generator state',
  '1104 recorded calls allGreedy are positive partial calls; future/global all-score deadness not inferred. Preserve order and all RNG effects'),
 ('CONFIDENCE_KEEP_POLICY',['M18','M19','M20','M21','M24'],
  ['raw draft hidden and ordered Markov seed+sample0..3','confidence/policy update'],['dynamic keep lengths','trimmed proposal retrieval','next iteration policy'],
  'iteration,raw versus normalized hidden,proposal order and old/new policy state',
  'fifth outputsample is not fifth Markov input; future keep cannot discard the head/hidden work needed to compute keep'),
 ('EXPERT_DISPATCH_AND_SHARED_OUTPUT',['M12','M13','M23','M24'],
  ['router/topK','peer dispatch','W8A8 expert FFNs','shared FFN','conditional EPLB map'],['receiver GEMMs','return/combine','routed/shared merge','original counters/state'],
  'EP participant,expert,valid received rows,credit/flag generation,quant rounds and scale',
  'existing F21 native fusion retained; sender0 can receive work; tile128 is not128 active rows. EPLBFalse snapshot does not remove other owned branch contracts'),
 ('GRAPH_NATIVE_PARAMETERS_WORKSPACE',['M00','M21','M22','M24'],
  ['capture/update/runtime setup','native workspace initialization'],['replay tasks','native asynchronous kernels','later update/free'],
  'task/handle/capture generation,address range,stream,event,last queued consumer',
  'private parameter mutability/lifetimes remain UNKNOWN. Preserve existing whole public op and state effects; no independence or direct private edit licensed'),
 ('COLLECTIVE_OWNER_AND_READINESS',['M07','M10','M11','M12','M13','M14','M16','M18','M19','M22','M23'],
  ['DP/TP/EP/CP participants and original producer writes'],['rank-local readers','peer receivers','collective output consumers','cross-stream waiters'],
  'group,rank,physical-card mapping,shard/range,collective epoch and publication flags',
  'retain participant counts and source protocol; raw link accounting is pool/topology-specific, not a universal wire floor or measured overlap'),
 ('OUTPUT_COMMIT_QUEUE_LIFETIME',['M01','M03','M17','M21','M24','M25'],
  ['accept/recover result','state commit','D2H completion','host queue publication'],['next input packing','cache rollback/recycle','client stream','stop accounting','buffer retirement'],
  'request generation,iteration,accepted count,token order,copy/event readiness',
  'preserve request cancellation/termination/output errors and untouched buffers; coalescing needs every queue consumer, not an inferred host critical path')
]
envelopes=[]
for sid,mods,writers,readers,ns,answer in states:
    envelopes.append({'id':sid,'modules':mods,'writer_ports':writers,'reader_ports':readers,
      'identity_and_epoch':ns,'conditional_fusion_split_or_reorder_answer':answer,
      'unknown_native_alias_or_unlisted_observer':'PRESERVE_INTERFACE_AND_ALL_ORIGINAL_EFFECTS;not proof of absence',
      'actual_all_native_consumers_or_complete_retirement_proved':False,
      'authority':[str(O/'module-phase-branch-matrix-v177.json'),str(N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json'),
                   str(N/'shared-profile-analysis-v176/public-bank-epoch-and-role-review.json'),str(H/'exact-guard-operator-contract-review.json')]})

save('host-score-and-state-port-contracts.json',{
  'status':'PUBLIC_BOUNDARY_CONTRACTS_AND_CONSERVATIVE_ALIAS_STATE_ENVELOPES',
  'pins':[pin(O/'module-phase-branch-matrix-v177.json'),pin(H/'hierarchical-G0-public-dependency-overlay.json')],
  'templates':templates,'state_envelopes':envelopes,
  'scope':'Fixed frozen text requests/config and applicable guarded operator interfaces. Admission/transport/error machinery is preserved as a public boundary; no framework redesign.',
  'not_native_alias_consumer_closure':True,'overall_complete':False})

g = read(H/'hierarchical-G0-public-dependency-overlay.json')
existing_templates = {i['template'] for i in g['instances']}
source_names = set()
for name in ['operator-graph-G5-v93.json','operator-graph-G6-v90.json','operator-graph-G7-v92.json','operator-graph-G8-v101.json','operator-graph-G9-v124.json']:
    source_names.update(t['id'] for t in read(O/name)['templates'])
boundary_names = {t['id'] for t in templates}
bindings=read(N/'shared-profile-analysis-v174/all42-composition-coverage.json')['candidate_bindings']
rows=[]
family_states = {
 'F01':['MHC_COEFFICIENT_RESIDUAL_AUX'], 'F02':['MHC_COEFFICIENT_RESIDUAL_AUX','REQUEST_INPUT_BANKS'],
 'F03':['MHC_COEFFICIENT_RESIDUAL_AUX','ENGRAM_HISTORY_AND_LOOKUP'],
 'F04':['MHC_COEFFICIENT_RESIDUAL_AUX','EXPERT_DISPATCH_AND_SHARED_OUTPUT'],
 'F05':['COLLECTIVE_OWNER_AND_READINESS'], 'F06':['COLLECTIVE_OWNER_AND_READINESS'],
 'F07':['POSITION_COUNT_SLOT_METADATA'], 'F08':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F09':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA','TOPK_SHARED_BANK'],
 'F10':['POSITION_COUNT_SLOT_METADATA','TOPK_SHARED_BANK'],
 'F11':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F12':['TOPK_SHARED_BANK','KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F13':['TOPK_SHARED_BANK','KV_RING_INDEX_CACHE','COLLECTIVE_OWNER_AND_READINESS'],
 'F14':['COLLECTIVE_OWNER_AND_READINESS','MHC_COEFFICIENT_RESIDUAL_AUX'],
 'F15':['COLLECTIVE_OWNER_AND_READINESS','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'F16':['COLLECTIVE_OWNER_AND_READINESS','MHC_COEFFICIENT_RESIDUAL_AUX'],
 'F17':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA','C2_BOOL_MASK_BANK'],
 'F18':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA','C2_BOOL_MASK_BANK'],
 'F19':['KV_RING_INDEX_CACHE','TOPK_SHARED_BANK'],
 'F20':['REQUEST_INPUT_BANKS','EXPERT_DISPATCH_AND_SHARED_OUTPUT','COLLECTIVE_OWNER_AND_READINESS'],
 'F21':['EXPERT_DISPATCH_AND_SHARED_OUTPUT','COLLECTIVE_OWNER_AND_READINESS','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'F22':['EXPERT_DISPATCH_AND_SHARED_OUTPUT','COLLECTIVE_OWNER_AND_READINESS'],
 'F23':['EXPERT_DISPATCH_AND_SHARED_OUTPUT','COLLECTIVE_OWNER_AND_READINESS','MHC_COEFFICIENT_RESIDUAL_AUX'],
 'F24':['ENGRAM_HISTORY_AND_LOOKUP','REQUEST_INPUT_BANKS','COLLECTIVE_OWNER_AND_READINESS'],
 'F25':['ENGRAM_HISTORY_AND_LOOKUP','MHC_COEFFICIENT_RESIDUAL_AUX'],
 'F26':['MHC_COEFFICIENT_RESIDUAL_AUX','TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS','SCORES_SAMPLING_RNG'],
 'F27':['MHC_COEFFICIENT_RESIDUAL_AUX','DRAFT_CONTEXT_SHARED_KV'],
 'F28':['REQUEST_INPUT_BANKS','MHC_COEFFICIENT_RESIDUAL_AUX','TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS','COLLECTIVE_OWNER_AND_READINESS'],
 'F29':['DRAFT_CONTEXT_SHARED_KV','KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F30':['MHC_COEFFICIENT_RESIDUAL_AUX','TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS','SCORES_SAMPLING_RNG','CONFIDENCE_KEEP_POLICY'],
 'F31':['POSITION_COUNT_SLOT_METADATA','TOPK_SHARED_BANK','C2_BOOL_MASK_BANK','DRAFT_CONTEXT_SHARED_KV','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'F32':['COLLECTIVE_OWNER_AND_READINESS','KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F33':['COLLECTIVE_OWNER_AND_READINESS','KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'F34':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA','C2_BOOL_MASK_BANK'],
 'F35':['SCORES_SAMPLING_RNG','TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS','OUTPUT_COMMIT_QUEUE_LIFETIME'],
 'H01':['OUTPUT_COMMIT_QUEUE_LIFETIME','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'H02':['REQUEST_INPUT_BANKS','POSITION_COUNT_SLOT_METADATA','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'H03':['KV_RING_INDEX_CACHE','POSITION_COUNT_SLOT_METADATA'],
 'H04':['REQUEST_INPUT_BANKS','OUTPUT_COMMIT_QUEUE_LIFETIME'],
 'H05':['OUTPUT_COMMIT_QUEUE_LIFETIME'],
 'H06':['REQUEST_INPUT_BANKS','POSITION_COUNT_SLOT_METADATA','COLLECTIVE_OWNER_AND_READINESS','GRAPH_NATIVE_PARAMETERS_WORKSPACE'],
 'H07':['REQUEST_INPUT_BANKS','TARGET_DRAFT_HEAD_AND_MARKOV_WEIGHTS','MHC_COEFFICIENT_RESIDUAL_AUX','COLLECTIVE_OWNER_AND_READINESS']}
assert set(family_states)=={b['candidate'] for b in bindings}
for b in bindings:
    assert all(t in source_names|boundary_names for t in b['templates'])
    rows.append({'candidate':b['candidate'],'templates':b['templates'],
      'instance_ids':[i['id'] for i in g['instances'] if i['template'] in b['templates']],
      'conditional_template_refs':[t for t in b['templates'] if t not in existing_templates],
      'state_envelope_refs':family_states[b['candidate']],
      'conditional_native_legality_not_promoted':True})
assert len(rows)==42 and len({r['candidate'] for r in rows})==42
covered_modules = set(m for t in templates for m in t['modules'])|set(m for s in envelopes for m in s['modules'])
assert covered_modules == {f'M{i:02}' for i in range(26)}
checks={'status':'PASS','public_boundary_templates':len(templates),'explicit_state_envelopes':len(envelopes),
        'all26_module_public_ports':len(covered_modules),'all42_family_template_refs_resolve':len(rows),
        'original_source_templates_retained':len(source_names),
        'all_native_consumers_aliases_retirement_proved':False,'whole_scope_complete':False}
save('candidate-to-public-hierarchy-bindings.json',{'status':'REFERENCE_BINDINGS_NOT_NATIVE_ATTRIBUTION','rows':rows,'checks':checks,
    'important_new_restrictions':[
      'Any F09/F10/F12/F31/F32/F33 and cross-layer/iteration composition that changes index generation/order must retain TopK per-range RAW/WAR/WAW epochs.',
      'F27/F28/F29/F30 use distinct target-context/query/raw-confidence public inputs; removed macro edge does not license their conflation.',
      'F24/F25 baseline has positive NPU host-UVA lookup and overlapFalse; an async-overlap schedule is a conditional alternative, not an already measured baseline.',
      'This resolves placeholder template names without treating every conditional native alias or observer as known.'
    ]})
save('boundary-state-port-checks.json',checks)

# Materialize the public boundary layer and stitch it to source/model ports.
# The graph remains a conditional effect graph, rather than a measured execution schedule.
combined=json.loads(json.dumps(g))
combined['nodes']=list(combined['nodes']); combined['edges']=list(combined['edges'])
ids={n['id'] for n in combined['nodes']}
def connect(a,b,v,kind='PUBLIC_PORT_BINDING',**kw):
    combined['edges'].append({'producer':a,'consumer':b,'value':v,'dependency_kind':kind,
                             'iteration_offset':0,**kw})
for t in templates:
    prefix='boundary.'+t['id']
    for n in t['nodes']:
        nid=prefix+'/'+n['id'];assert nid not in ids;ids.add(nid)
        combined['nodes'].append({'id':nid,**{k:v for k,v in n.items() if k!='id'},
                                  'modules':t['modules'],'template':t['id'],'accounting_level':'public_boundary'})
    for e in t['edges']:
        connect(prefix+'/'+e['producer'],prefix+'/'+e['consumer'],e['value'],e['dependency_kind'])
def B(t,n):return 'boundary.'+t+'/'+n
joins=[
 (B('HOST_TOKENIZE','tokens'),B('HOST_PREFIX_HASH','tokens'),'frozen tokenized text'),
 (B('HOST_TOKENIZE','resolved_sampling_contract'),B('SCORE_SELECTION_AND_REJECTION','sampling_metadata'),'original resolved request sampling contract'),
 (B('HOST_PREFIX_HASH','page_leases'),B('METADATA_AND_STATE_PUBLICATION','page_leases'),'prefix/KV leases with original ownership/generation'),
 (B('HOST_INPUT_PACK','row_map'),B('METADATA_AND_STATE_PUBLICATION','row_map'),'same row map'),
 (B('HOST_INPUT_PACK','counts'),B('METADATA_AND_STATE_PUBLICATION','counts'),'same valid and padded counts'),
 (B('HOST_INPUT_PACK','positions'),B('METADATA_AND_STATE_PUBLICATION','positions'),'same token/query positions'),
 (B('HOST_INPUT_PACK','slots'),B('METADATA_AND_STATE_PUBLICATION','slots'),'same physical slots'),
 (B('HOST_INPUT_PACK','packed_input'),B('GRAPH_INPUT_BANK','host_staging'),'packed original input values'),
 (B('METADATA_AND_STATE_PUBLICATION','state_publication'),B('GRAPH_INPUT_BANK','metadata'),'published state generation/readiness'),
 (B('GRAPH_INPUT_BANK','bank_new'),'inputs','public model input bank for selected call'),
 (B('METADATA_AND_STATE_PUBLICATION','attention_metadata'),'metadata','selected attention metadata public port'),
 (B('METADATA_AND_STATE_PUBLICATION','draft_metadata'),'draft.pipeline/draft_input_metadata','draft expansion/context/query mapping'),
 (B('SCORE_SELECTION_AND_REJECTION','accepted_ids'),B('HOST_INPUT_PACK','previous_sampled_ids'),'accepted IDs feed next packing'),
 (B('SCORE_SELECTION_AND_REJECTION','output_payload'),B('HOST_RESPONSE_READY','accepted_device_results'),'same token/score payload'),
 (B('HOST_RESPONSE_READY','d2h_ready'),B('HOST_COMPLETION_NOTIFY','device_completion'),'completion/data readiness pair'),
 (B('SCORE_SELECTION_AND_REJECTION','accepted_lengths'),B('METADATA_AND_STATE_PUBLICATION','accepted_lengths_old'),'next iteration committed prefix'),
 ('draft.pipeline/draft_token_output',B('SCORE_SELECTION_AND_REJECTION','draft_ids'),'original proposal order before verification'),
 ('draft.pipeline/probability_output',B('SCORE_SELECTION_AND_REJECTION','draft_probabilities'),'conditional proposal probabilities, never inferred mandatory'),
 ('draft.pipeline/sampling_metadata',B('SCORE_SELECTION_AND_REJECTION','sampling_metadata'),'same resolved request rows and generator contract')]
for a,b,v in joins:
    kw={}
    if 'next ' in v:kw['iteration_offset']=1
    connect(a,b,v,**kw)
connect('target.final',B('SCORE_SELECTION_AND_REJECTION','target_head'),
        'public head subregion of the collapse/norm/head macro; same computation, not an extra head', 'ABSTRACTION_BINDING')
connect(B('SCORE_SELECTION_AND_REJECTION','verify_recover'),'verify_accept',
        'same sampler/rejection public composite, with current source mapping', 'ABSTRACTION_BINDING')
for n in combined['nodes']:
    if n['id']=='verify_accept':
        n['historical_source']=n.get('source');n['source']='pinned selected sampler/rejection/proposer functions in v177 census; v176 recorded1104 scalar bindings'
        n['subgraph_status']='PUBLIC_SOURCE_AND_CONDITIONAL_VARIANTS_BOUND;PRIVATE_NATIVE_INTERNALS_NOT_CLAIMED'
for i in range(40):
    nid=f'target.layers.{i}.attention.detail/index_metadata'
    if nid in ids:connect(B('METADATA_AND_STATE_PUBLICATION','index_metadata'),nid,'source-layer index metadata, lengths and candidate-generation contract')
connect(B('GRAPH_INPUT_BANK','replay_or_eager'),'embedding','selected execution/control branch; not an additional arithmetic producer','CONTROL_INTERFACE')
connect(B('HOST_RESPONSE_READY','release'),B('HOST_PREFIX_HASH','retirement'),
        'request/buffer/page readers retired before recycling','STATE_RETIREMENT',iteration_offset=1)

# Engram lookup is prepared globally and rows are consumed at two different layers.
# A single parameterized source template avoids inventing two global hashes/lookups.
engram=next(t for t in read(O/'operator-graph-G5-v93.json')['templates'] if t['id']=='ENGRAM')
for n in engram['nodes']:
    nid='engram.public/'+n['id'];assert nid not in ids;ids.add(nid)
    combined['nodes'].append({'id':nid,**{k:v for k,v in n.items() if k!='id'},
                              'modules':['M14','M21','M23','M24'],'template':'ENGRAM',
                              'case_parameters':{'consuming_layer':[1,14]},
                              'accounting_level':'parameterized_refinement_not_duplicate_global_lookup'})
for e in engram['edges']:
    connect('engram.public/'+e['producer'],'engram.public/'+e['consumer'],e['value'],e['dependency_kind'],condition=e.get('condition'))
connect('inputs','engram.public/tokens','original input/lookback IDs')
connect('metadata','engram.public/hash_runtime_metadata','valid count/positions/token selection/hash metadata')
connect('engram.public/lookup','engram_lookup','same prepared lookup public region','ABSTRACTION_BINDING')
for i,slot in [(1,0),(14,1)]:
    connect(f'target.layers.{i-1}.hc_ffn_post','engram.public/residual',
            'per-consuming-layer original HC residual; cases differ','PARAMETERIZED_PORT_BINDING',condition=f'consuming_layer=={i}')
    connect('engram.public/gate_result',f'target.layers.{i}.engram',
            f'layer{i} selected Engram row slot{slot}; not a common gate result across layers','PARAMETERIZED_PORT_BINDING',condition=f'consuming_layer=={i}')

unbound=sorted({e[k] for e in combined['edges'] for k in ['producer','consumer'] if e[k] not in ids})
assert not unbound,unbound
combined.update(status='HIERARCHICAL_PUBLIC_CONDITIONAL_OPERATOR_AND_STATE_GRAPH',
    boundary_contracts=pin(H/'host-score-and-state-port-contracts.json'),
    candidate_bindings=pin(H/'candidate-to-public-hierarchy-bindings.json'),
    source_template_definitions=[{'source':pin(O/name),'template_ids':[t['id'] for t in read(O/name)['templates']]}
        for name in ['operator-graph-G5-v93.json','operator-graph-G6-v90.json','operator-graph-G7-v92.json','operator-graph-G8-v101.json','operator-graph-G9-v124.json']],
    parameterized_source_bindings=[
      {'template':'MHC_BLOCK_AND_CROSS_LAYER','case_domain':'target.layers0..39 and applicable inherited draft.layers40..42',
       'ports':'residual,pre_mix,HCpre/post,norm/cast,attention,MoE,aux,Engram,collapse/finalnorm; retain macro/source distinction'},
      {'template':'MHC_ATTENTION_COEFFICIENT_FORK','case_domain':'conditional documented split of HC pre/attention coefficient branches',
       'ports':'all coefficient/residual/numerical outputs retained; no native handoff proved'},
      {'templates':['SAMPLER_GREEDY_FALLBACK_ROUTE','SAMPLER_GREEDY_GENERAL_BODY','SAMPLER_GREEDY_ACTUAL_N1_BODY'],
       'case_domain':'selected greedy branch; actual n and RNG path conditional, not inferred from output width'},
      {'templates':['ASCEND_DEFAULT_PROBABILITY_CUTOFF_AND_RACE','UPSTREAM_MONOLITHIC_PIVOT_FILTER',
                    'RECOVERY_PYTORCH_OWNER_MAP','RECOVERY_PYTORCH_VALUES','VERIFY_PYTORCH_ORDINARY','VERIFY_PYTORCH_BLOCK','RECOVERY_OPTIONAL_BLOCKWISE'],
       'case_domain':'random/mixed/recovery alternative public contracts retained; partial1104 current scalar calls do not execute these cases'},
      {'templates':['PREPARE_VISIBLE_MULTISET','DSPARK_LOGICAL_WINDOW_BROADCAST','SMLA_PA_INDEX_OCCURRENCES'],
       'case_domain':'metadata/index occurrence producers and consumers under original ownership and alias epochs'}],
    remaining_public_graph_work=[
      'Independently check exact consumer/alias/state envelope adequacy against retained profile/source effect evidence; endpoint resolution alone is insufficient.',
      'Run declared rules over the corrected boundary and shared-bank interactions. Do not reopen Cartesian candidate pairs.'
    ],whole_scope_complete=False)
save('hierarchical-G0-with-boundary-and-state-ports.json',combined)
save('assembled-public-G0-checks.json',{'status':'PASS','nodes':len(combined['nodes']),'edges':len(combined['edges']),
     'unbound_endpoint_ids':unbound,'all42_family_refs_resolved':True,'native_schedule_or_all_alias_consumers_proved':False,
     'whole_scope_complete':False})
print(json.dumps(checks,ensure_ascii=False))
