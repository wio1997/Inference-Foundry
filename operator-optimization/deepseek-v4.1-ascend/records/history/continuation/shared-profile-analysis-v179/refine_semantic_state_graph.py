"""Repair public effects found by reviewing the existing360semantic contracts."""
import copy
import hashlib
import json
from pathlib import Path
N=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030');O=N/'profile-first-20261031';H=N/'shared-profile-analysis-v179';P=N/'shared-profile-analysis-v178'
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
g=copy.deepcopy(read(P/'hierarchical-G0-with-boundary-and-state-ports.json'))
ports=copy.deepcopy(read(P/'host-score-and-state-port-contracts.json'))
bindings=copy.deepcopy(read(P/'candidate-to-public-hierarchy-bindings.json'))
contract=read(O/'index-context-operator-contract-v64.json')
assert any('KV3distinct5120' in s for s in contract['facts'])
witness=read(O/'heads-sampling-operator-contract-v65.json')['witnesses']['snapshot_alias_race']
old=witness['old_computed'];idx=witness['previous_positions'];valid=witness['valid']
snapshot=[old[idx[i]]+valid[i] for i in range(len(old))]
inplace=list(old)
for i in range(len(old)):inplace[i]=inplace[idx[i]]+valid[i]
assert snapshot==witness['required']==[21,11] and inplace==witness['naive_inplace']==[21,22]

retired_nodes=[n for n in g['nodes'] if n['id'].startswith('draft.shared_kv/')]
retired_edges=[e for e in g['edges'] if any(e[k].startswith('draft.shared_kv/') for k in ['producer','consumer'])]
g['nodes']=[n for n in g['nodes'] if not n['id'].startswith('draft.shared_kv/')]
g['edges']=[e for e in g['edges'] if not any(e[k].startswith('draft.shared_kv/') for k in ['producer','consumer'])]
g['instances']=[i for i in g['instances'] if i['id']!='draft.shared_kv']
ids={n['id'] for n in g['nodes']}
def node(id,kind,mods,**kw):
    assert id not in ids,id
    ids.add(id);g['nodes'].append({'id':id,'kind':kind,'modules':mods,**kw})
def edge(a,b,v,kind='PUBLIC_CONTRACT_RAW',**kw):
    g['edges'].append({'producer':a,'consumer':b,'value':v,'dependency_kind':kind,'iteration_offset':0,**kw})
tpl=next(t for t in read(O/'operator-graph-G5-v93.json')['templates'] if t['id']=='DSPARK_SHARED_KV')
for j,layer in enumerate([40,41,42]):
    prefix=f'draft.shared_kv.layer{layer}'
    for n in tpl['nodes']:
        node(prefix+'/'+n['id'],n.get('kind','PUBLIC_PORT'),['M18','M19','M21','M24'],
             source_template='DSPARK_SHARED_KV',source_local_node=n['id'],
             owner_layer=layer,parameter_namespace=f'draft.model.layers.{layer}.self_attn',
             cache_group_and_slot_bank='selected per-layer group;not inferred equal to another layer',
             accounting_level='refinement of three-element context-KV collection')
    for e in tpl['edges']:
        edge(prefix+'/'+e['producer'],prefix+'/'+e['consumer'],e['value'],e['dependency_kind'],
             owner_layer=layer,ancestry={'template':'DSPARK_SHARED_KV','local_edge':e})
    edge('draft.pipeline/context_buffer',prefix+'/context',
         'common normalized context-bank content;preserve original prefix copy/tail/last readers')
    edge('draft.pipeline/context_slots',prefix+'/slots',
         'select this layer group slot mapping;different groups can have different banks',
         condition='per-layer group selector;None/empty store guard originally follows projection')
    edge('draft.pipeline/context_positions',prefix+'/context_positions',
         'same logical context positions,per-layer rotary configuration/rounds retained')
    edge(prefix+'/persistent_swa_state',f'draft.pipeline/block{j}',
         f'layer{layer} context KV cache and its valid generation;not another layer KV values',
         'LAYER_OWNED_PERSISTENT_STATE_READ')
    g['instances'].append({'id':prefix,'template':'DSPARK_SHARED_KV','parent_macro':'dspark.main',
       'owner_layer':layer,'accounting':'one of3layer-specific contractions/norms/cache writes;common context input counted once',
       'native_lowering_or_instance_order_proved':False})

# G5's singular names are collection macros. Do not let their old fanout imply equal layer values.
old_summary_edges=[]
newedges=[]
for e in g['edges']:
    if e['producer']=='draft.pipeline/kv_state' and e['consumer'] in ['draft.pipeline/block0','draft.pipeline/block1','draft.pipeline/block2']:
        old_summary_edges.append(e)
    else:newedges.append(e)
g['edges']=newedges
for n in g['nodes']:
    if n['id'] in ['draft.pipeline/kv_precompute','draft.pipeline/kv_state']:
        n['historical_kind']=n.get('kind');n['kind']='COLLECTION_OF3_LAYER_SPECIFIC_KV_OPERATIONS_OR_STATES'
        n['does_not_assert_identical_KV_values']=True
for layer in [40,41,42]:
    edge(f'draft.shared_kv.layer{layer}/persistent_swa_state','draft.pipeline/kv_state',
         'one distinct layer cache in collection;abstraction identity,not another copy','ABSTRACTION_BINDING')

# Candidate2048 block IDs are distinct from per-layer512 position TopK values.
obsolete_candidate_edges=[]
newedges=[]
for e in g['edges']:
    if e['producer']=='target.layers.20.attention' and e['consumer'].endswith('/candidate_state_and_lengths'):
        obsolete_candidate_edges.append(e)
    else:newedges.append(e)
g['edges']=newedges
assert len(obsolete_candidate_edges)==4
node('state.candidate_block_bank','VERSIONED_CANDIDATE_BLOCK_ID_BANK',['M09','M10','M21','M22','M24'],
     logical_id_width='INT32',logical_ids_per_query=2048,
     allocated_shape_bytes='UNKNOWN;do not infer from per-query schema or16MiBTopK backing',
     namespace=['worker/device','candidate source20','request/query row','allocation generation','visible block/length epoch'])
node('target.layers.20.candidate_raw_out','FRESH_PUBLIC_CANDIDATE_OUTPUT',['M09','M21'],
     interface_contract='mode1candidate_out is fresh,then copied to bank;mode2candidate argument is input-only',
     evidence=str(O/'index-context-operator-contract-v64.json'))
node('target.layers.20.candidate_publication','PUBLIC_BANK_COPY_OR_CONDITIONAL_NEW_OUT_ABI',['M09','M21','M22','M24'])
edge('target.layers.20.attention','target.layers.20.candidate_raw_out',
     'mode1 candidate output subport;separate from512-position indices','ABSTRACTION_OUTPUT_PORT')
edge('target.layers.20.candidate_raw_out','target.layers.20.candidate_publication',
     'baseline output-to-bank copy;elimination requires newoutABI,not prepare-indices self-copy identity')
edge('target.layers.20.candidate_publication','state.candidate_block_bank','candidate2048 IDs,valid lengths and latest-visible block contract','BANK_WRITE_EFFECT')
for layer in [24,28,32,36]:
    port=f'target.layers.{layer}.attention.detail/candidate_state_and_lengths'
    edge('state.candidate_block_bank',port,'mode2 membership filter input from source20;outside-candidate finite fillers/ties preserved','CANDIDATE_EPOCH_RAW')
    edge(f'target.layers.{layer}.attention.detail/qli','target.layers.20.candidate_publication',
         'last overlapping candidate read before next-iteration overwrite','CANDIDATE_RANGE_WAR',iteration_offset=1,
         condition='same physical allocation and overlapping row range;actual native queued read must retire')
edge('target.layers.20.candidate_publication','target.layers.20.candidate_publication',
     'candidate epoch advances independently of TopK position epochs','CANDIDATE_RANGE_WAW',iteration_offset=1)

# Three gathers precede the first write. The accepted ELSE read occurs AFTER the computed write.
# A snapshot variant may hoist that extra read only after excluding/preserving cross-plane alias effects.
batch_nodes=['old_computed','accepted_state','previous_positions','valid_sampled_counts','prev_num_draft_tokens',
             'cpu_num_computed_tokens','gather_indices','gathered_valid_counts','gathered_prev_computed','gathered_prev_drafts',
             'participation','gather_snapshot','new_computed','computed_commit','accepted_else_read',
             'new_accepted_counts','accepted_commit','next_generation','valid_count_summary']
for name in batch_nodes:
    node('batch_correction/'+name,'BATCH_OLD_SNAPSHOT_OR_NEW_GENERATION_PORT',['M04','M21','M24'],
         namespace='request generation,old remap,state plane,row map,int width and original sentinel/error semantics')
edge('batch_correction/previous_positions','batch_correction/gather_indices','clamp_min0;original bounds/errors still apply to all3gathers')
for source,target in [('valid_sampled_counts','gathered_valid_counts'),('old_computed','gathered_prev_computed'),('prev_num_draft_tokens','gathered_prev_drafts')]:
    edge('batch_correction/'+source,'batch_correction/'+target,'original advanced-index gather,including nonparticipating rows')
    edge('batch_correction/gather_indices','batch_correction/'+target,'same clamped gather indices')
    edge('batch_correction/'+target,'batch_correction/gather_snapshot','all3old gathered arrays before computed writes','SNAPSHOT_READSET')
edge('batch_correction/previous_positions','batch_correction/participation','prev_positions>=0')
edge('batch_correction/gathered_prev_drafts','batch_correction/participation','prev_drafts>0;ANDoriginalpredicate')
edge('batch_correction/gather_snapshot','batch_correction/new_computed','prev_computed+gatheredvalid.int32;original overflow and shapes')
edge('batch_correction/participation','batch_correction/new_computed','where(participating,corrected,cpucomputed)')
edge('batch_correction/cpu_num_computed_tokens','batch_correction/new_computed','CPU fallback tensor values before original computed copy')
edge('batch_correction/new_computed','batch_correction/computed_commit','first original destination write;num_computed[:n]INT32','CROSS_ROW_WAR')
edge('batch_correction/computed_commit','batch_correction/accepted_else_read','baseline accepted ELSE values are read after computed copy;preserve any cross-plane alias effect','ORDERED_POTENTIAL_ALIAS_READ_AFTER_WRITE')
edge('batch_correction/accepted_state','batch_correction/accepted_else_read','accepted tensor at original read epoch,not automatically an earlier oldsnapshot')
edge('batch_correction/accepted_else_read','batch_correction/new_accepted_counts','where(participating,gatheredvalid,accepted_else)')
edge('batch_correction/gathered_valid_counts','batch_correction/new_accepted_counts','original INT64gather,not reread after mutation')
edge('batch_correction/participation','batch_correction/new_accepted_counts','same participation mask')
edge('batch_correction/new_accepted_counts','batch_correction/accepted_commit','second destinationwrite;original promotedwhere then copyINT32')
edge('batch_correction/accepted_commit','batch_correction/next_generation','coherent next generation after both destination writes')
edge('batch_correction/next_generation','boundary.HOST_INPUT_PACK/request_generation','next packing uses committed batch generation','PUBLIC_PORT_BINDING',iteration_offset=1)
edge('boundary.SCORE_SELECTION_AND_REJECTION/accepted_ids','batch_correction/valid_count_summary','INT64summaryofliteral(x!=-1)&(x<V);includesnegative<-1unlessvaliddomainproved')
edge('batch_correction/valid_count_summary','batch_correction/valid_sampled_counts','published prior sampling valid-count generation','PUBLIC_PORT_BINDING',iteration_offset=1)

additional=[
 {'id':'CANDIDATE_BLOCK_BANK','modules':['M09','M10','M21','M22','M24'],
  'writer_ports':['mode1 source20fresh candidate output','output-to-bank copy or separatelyproved newoutABI'],
  'reader_ports':['mode2source24/28/32/36 membership filters','all original length/latest-visible and graph consumers'],
  'identity_and_epoch':'candidate2048block IDs separate from512position TopK;worker/device/request/query/visible-block/allocation generation',
  'conditional_fusion_split_or_reorder_answer':'Retain mode1fresh output contract,copy/newABI and4readers. Bitmap544B/query vs8192Blogical IDs is conditional representation;preserve latest-visible pin,outside finite fillers,ties and boundaries;not measuredHBM/gain.',
  'unknown_native_alias_or_unlisted_observer':'PRESERVE_ALL_ORIGINAL_EFFECTS;no all-private-alias proof or allocation-size claim',
  'actual_all_native_consumers_or_complete_retirement_proved':False,'authority':[str(O/'index-context-operator-contract-v64.json')]},
 {'id':'BATCH_CORRECTION_SNAPSHOT','modules':['M04','M21','M24'],
  'writer_ports':['num_computed_tokens[:n]INT32firstwrite','num_accepted_tokensINT32secondwrite','new request/remap generation'],
  'reader_ports':['3oldgathersbeforecomputedwrite','CPUcomputedfallbackbeforefirstwrite','acceptedELSEreadAFTERcomputedwrite','next input/position packing','asynchronous original consumers'],
  'identity_and_epoch':'3gather snapshot plus original per-plane read epoch;acceptedELSE cannot be blindly hoisted across computedwrite if planesalias',
  'conditional_fusion_split_or_reorder_answer':'SmallNone-program gathered snapshot or immutableoldstate can fuse updates. Per-row inplace gather-map is incorrect:[1,0] gives[21,22] instead of[21,11]. Snapshot-all-sources needs disjointplanes or explicit emulation of original computedwrite->acceptedELSEread. Preserve CPUfallback,int32overflow,promotedwhere/copy,negative/sentinel/bounds/error and queuedreader semantics.',
  'unknown_native_alias_or_unlisted_observer':'PRESERVE_ALL_ORIGINAL_EFFECTS;actual pointer aliases require binding only for stronger private elision',
  'actual_all_native_consumers_or_complete_retirement_proved':False,'authority':[str(O/'heads-sampling-operator-contract-v65.json')]},
 {'id':'SP_VALIDITY_MASK','modules':['M05','M11','M12','M14','M19','M21','M22','M23'],
  'writer_ports':['None-mask false-prefix initialization','provided-mask copy','Truepadding tail'],
  'reader_ports':['SPpacking/sharding','selected attention/MoE/Engram/collective valid-row readers','graphbank consumers'],
  'identity_and_epoch':'mask purpose,actualprefix values,provided-vsNone branch,N/Pandrequest/capture generation;not C2mask bank identity',
  'conditional_fusion_split_or_reorder_answer':'Full-write public pad replacement may combine prefix/tail stores while preserving suppliedTrueprefix,clone/noalias semantics and allvalid-row readers. BOOLshape or old16680records cannot identify current171C2/SP masks.',
  'unknown_native_alias_or_unlisted_observer':'Keep native helper/workspace effects unless whole public call is equivalently replaced',
  'actual_all_native_consumers_or_complete_retirement_proved':False,'authority':[str(O/'embedding-initialization-operator-contract-v67.json'),str(N/'shared-profile-analysis-v176/public-bank-epoch-and-role-review.json')]}
]
ports['state_envelopes'].extend(additional)
for s in ports['state_envelopes']:
    if s['id']=='DRAFT_CONTEXT_SHARED_KV':
        s['id']='DRAFT_CONTEXT_AND_LAYER_OWNED_KV'
        s['writer_ports']=['common ordered target aux concat/proj/full5120norm/contextbank','3layer-specific WKV+norm/RoPE/scatter']
        s['reader_ports']=['layer40query->layer40KV','layer41query->layer41KV','layer42query->layer42KV','futureeligible generation readers']
        s['conditional_fusion_split_or_reorder_answer']='Share common context loads or joint1536projection input,retain3distinct512outputs,3norms,per-layerparameters/rotary/cache-group/slots and first-layer readiness. Common context/shared method name is not equal KV results;original guards/store ordering preserved.'
        s['authority'].append(str(O/'index-context-operator-contract-v64.json'))
    if s['id']=='KV_RING_INDEX_CACHE':
        s['conditional_fusion_split_or_reorder_answer']+=' Full32/private16/paired compact-owner alternatives retain SWAanchor/globalID reservation,half-owner lifetime,ABA generation,lastqueuedwriter and oldpoolread-before-update;not universalhalf-capacity saving.'
    if s['id']=='ENGRAM_HISTORY_AND_LOOKUP':
        s['conditional_fusion_split_or_reorder_answer']+=' Inactivegate is originalhidden+0*value,not unconditional identity for NaN/Inf/signbit. Keep originalunrotated residual and fullHnorm/rounds when pruning key/dot branches.'
    if s['id']=='REQUEST_INPUT_BANKS':
        s['conditional_fusion_split_or_reorder_answer']+=' Source negative advanced indices wrap but destination negatives/errors differ;duplicates/sample-then-draft priority and oldsourcealias snapshots survive combined scatters. Batchcorrection needs all3gathers beforefirstwrite plus original computedwrite->acceptedELSEread order if cross-planealiases,not per-row ordering.'

for row in bindings['rows']:
    row['state_envelope_refs']=['DRAFT_CONTEXT_AND_LAYER_OWNED_KV' if s=='DRAFT_CONTEXT_SHARED_KV' else s for s in row['state_envelope_refs']]
    if row['candidate'] in ['F09','F10','F12','F13','F19','F31']:
        row['state_envelope_refs'].append('CANDIDATE_BLOCK_BANK')
    if row['candidate'] in ['F20','F31','H02','H06','H07']:
        row['state_envelope_refs'].append('BATCH_CORRECTION_SNAPSHOT')
    if row['candidate'] in ['F15','F20','F24','F28','F31','H02','H06','H07']:
        row['state_envelope_refs'].append('SP_VALIDITY_MASK')
    if row['candidate']=='F29':
        row['instance_ids']=[f'draft.shared_kv.layer{i}' for i in [40,41,42]]
        row['common_input_is_not_common_output']=True

g.update(status='SEMANTICALLY_REFINED_PUBLIC_CONDITIONAL_OPERATOR_GRAPH',
    parent=pin(P/'hierarchical-G0-with-boundary-and-state-ports.json'),
    state_contracts=str(H/'host-score-and-state-port-contracts.json'),
    candidate_bindings=str(H/'candidate-to-public-hierarchy-bindings.json'),
    corrections=['3layer-owned DSpark context KV outputs/caches','separate freshcandidate2048 output/bank/readers','batch3gather snapshot and ordered potential cross-planealias read','SPvaliditymask separate from C2mask'],
    removed_coarse_edges={'shared_KV':retired_edges,'summary_collection_fanout':old_summary_edges,'candidate_publication':obsolete_candidate_edges},
    remaining_public_graph_work=['Requirement-by-requirement completion audit must evaluate the semantic review and retained conditional gates;no graph counts as standalone completion proof.'],
    whole_scope_complete=False)
ports.update(parent=pin(P/'host-score-and-state-port-contracts.json'),status='REFINED_PUBLIC_CONSUMER_ALIAS_AND_STATE_CONTRACTS',overall_complete=False)
bindings.update(parent=pin(P/'candidate-to-public-hierarchy-bindings.json'),status='REFINED42FAMILY_STATE_AND_OWNER_BINDINGS')
bindings['checks']={'status':'PASS','all42_refs_resolve':True,'semantic_refinements':g['corrections'],'whole_scope_complete':False}
for p,x in [('hierarchical-G0-with-boundary-and-state-ports.json',g),('host-score-and-state-port-contracts.json',ports),('candidate-to-public-hierarchy-bindings.json',bindings)]:save(p,x)
assert len(retired_nodes)==9 and len(g['instances'])==124
assert len(ids)==len(g['nodes'])
assert all(e[k] in ids for e in g['edges'] for k in ['producer','consumer'])
assert len(ports['state_envelopes'])==18
assert all(set(b['state_envelope_refs'])<={s['id'] for s in ports['state_envelopes']} for b in bindings['rows'])
checks={'status':'PASS','nodes':len(g['nodes']),'edges':len(g['edges']),'materialized_model_template_instances':len(g['instances']),
 'context_KV_layer_namespaces':[40,41,42],'candidate_filter_reader_layers':[24,28,32,36],
 'state_envelopes':len(ports['state_envelopes']),'batch_snapshot_witness':{'required':snapshot,'naive_inplace':inplace},
 'original360_semantic_rows_preserved':True,'no_new_NPU_model_restart':True,'overall_complete':False}
save('semantic-graph-refinement-checks.json',checks)
ref=next(s for s in read(O/'heads-sampling-operator-contract-v65.json')['source_files'] if s['path'].endswith('/spec_decode/utils.py'))
save('targeted-batch-source-order-review.json',{
 'status':'PINNED26LINE_OPERATOR_UPDATE_BODY_REVIEWED_FOR_ALIAS_ORDER',
 'source':ref,'symbol':'update_num_computed_tokens_for_batch_change','first_line':61,'last_line':86,
 'reason':'Profileproves3gathers/2writes,butnotacceptedELSEexpressionevaluationorder/crossplanealiases',
 'facts':['3advancedindexgathers before num_computed[:n].copy_',
          'corrected=prev_computed+valid_counts.int32;participating=prev_positions>=0ANDprev_drafts>0',
          'CPUfallbackwhere evaluatedbeforecomputedcopy',
          'num_acceptedwhereELSEreadsacceptedaftercomputedcopy;thenoriginaldtypepromotion/copy',
          'alloldsource snapshot across2planes is conditional on no changed computed->acceptedalias effect'],
 'new_remote_source_or_NPU_work':False,'whole_scope_complete':False})
print(json.dumps(checks,ensure_ascii=False))
