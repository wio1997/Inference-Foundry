"""Offline reconciliation of existing profile/source evidence; no target execution."""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path

N = Path('/Users/wio/work/166-vllm-analysis-continuation-20261030')
O = N / 'profile-first-20261031'
H = N / 'shared-profile-analysis-v178'
H177 = N / 'shared-profile-analysis-v177'

def read(p):
    return json.loads(p.read_text())

def pin(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def save(name, d):
    (H / name).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

cfgp = N/'sampling-guard-source-v55/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json'
srcp = N/'original-source-cache-v53/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
assert pin(cfgp)['sha256'] == '43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
assert pin(srcp)['sha256'] == '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'
cfg = read(cfgp)
cfg = cfg.get('text_config', cfg)
tree = ast.parse(srcp.read_text())
defs = {x.name: x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
plan_refs = [{'symbol': k, 'first': defs[k].lineno, 'last': defs[k].end_lineno,
              'source_sha256': pin(srcp)['sha256']}
             for k in ['_latest_source', 'build_layer_plan']]
def latest(i, sources):
    return next((s for s in reversed(sources) if s <= i), None)

roles = []
for i, ratio in enumerate(cfg['compress_ratios'][:cfg['num_hidden_layers']]):
    kv = latest(i, cfg['kv_source_layer_ids']) if ratio else None
    ix = latest(i, cfg['index_source_layer_ids']) if ratio else None
    is_kv = i in cfg['kv_source_layer_ids']
    is_ix = i in cfg['index_source_layer_ids']
    template = ('ATTN_SWA' if not ratio else
                'ATTN_C2_SOURCE' if ratio == 2 and is_ix else
                'ATTN_C2_CONSUMER' if ratio == 2 else
                'ATTN_C1_SOURCE' if is_kv else
                'ATTN_C1_INDEX_SOURCE' if is_ix else 'ATTN_C1_CONSUMER')
    roles.append({'layer': i, 'compress_ratio': ratio,
                  'kv_source_layer': kv, 'index_source_layer': ix,
                  'is_kv_source': is_kv, 'is_index_source': is_ix,
                  'is_candidate_source': i == cfg['candidate_source_layer_id'],
                  'uses_candidate_filter': ix is not None and ix > cfg['candidate_source_layer_id'],
                  'engram_slot': cfg['engram_layer_ids'].index(i) if i in cfg['engram_layer_ids'] else None,
                  'attention_template': template})
assert len(roles) == 40
assert collections.Counter(r['attention_template'] for r in roles) == {
    'ATTN_SWA': 2, 'ATTN_C2_SOURCE': 3, 'ATTN_C2_CONSUMER': 15,
    'ATTN_C1_SOURCE': 1, 'ATTN_C1_INDEX_SOURCE': 4, 'ATTN_C1_CONSUMER': 15}

catalog = read(H177/'loaded-module-instance-census.json')
selected = []
selector_names = {'DeepseekV41Model', 'DeepseekV41MoE', 'AscendMoERunner',
                  'AscendRoutedExperts', 'AscendParallelEngramEmbedding',
                  'AscendParallelLMHead', 'DeepseekV41Compressor', 'DeepseekV41Indexer'}
for cls in catalog['classes']:
    if cls['qualified_class'].split('.')[-1] in selector_names:
        selected.append(copy.deepcopy(cls))
norms = [i for c in catalog['classes'] if c['qualified_class'].endswith('.AscendRMSNorm') for i in c['instances']]
assert len(norms) == 582 and all(i['scalar_attributes'].get('bias_loaded') is False for i in norms)
models = [i for c in catalog['classes'] if c['qualified_class'].endswith('.DeepseekV41Model') for i in c['instances']]
assert len(models) == 2
assert all(i['scalar_attributes'].get('_engram_overlap_enabled') is False for i in models)
assert all(i['scalar_attributes'].get('engram_rotated') is True for i in models)
binding = {
    'status': 'FIXED_CONFIG_LAYER_ROLES_AND_POSITIVE_CATALOGUE_SELECTORS',
    'pins': [pin(cfgp), pin(srcp), pin(H177/'loaded-module-instance-census.json')],
    'targeted_source_functions': plan_refs, 'target_layers': roles,
    'role_counts': dict(collections.Counter(r['attention_template'] for r in roles)),
    'draft_layer_ids': [40, 41, 42],
    'draft_G1_aliases': {'dspark.blocks.0': 40, 'dspark.blocks.1': 41, 'dspark.blocks.2': 42},
    'selected_catalogue_classes': selected,
    'bias_snapshot': {'catalogue_entries': len(norms), 'all_bias_loaded_false': True},
    'engram_snapshot': {'target_model_entries':len(models), 'all_overlap_false':True,'all_rotated_true':True},
    'limits': [
        'Catalogue entries include capture/formal snapshots of EP0; they are not 582 independent runtime calls.',
        'Layer roles come from pinned config/build_layer_plan, not the attention wrapper compress_ratio scalar.',
        'Lookup dp_size=1 does not imply global DP=1 or remove host-shared/DP hash ports.',
        'No zero bias, no module hook and no observed branch do not prove every other observer absent.',
        'Pointer identity does not imply numerical equality or a constant value epoch.'
    ]}
save('layer-role-and-selector-bindings.json', binding)

graph_paths = [O/'operator-graph-G1-v88.json', O/'operator-graph-G5-v93.json',
               O/'operator-graph-G6-v90.json', O/'operator-graph-G7-v92.json',
               O/'operator-graph-G8-v101.json', O/'operator-graph-G9-v124.json']
g1 = read(graph_paths[0]); templates = {}
for p in graph_paths[1:]:
    for t in read(p)['templates']:
        assert t['id'] not in templates
        templates[t['id']] = copy.deepcopy(t)

nodes = copy.deepcopy(g1['nodes'])
node_ids = {n['id'] for n in nodes}
edges = []
removed = []
for j, e in enumerate(g1['edges']):
    if e['producer'] == 'dspark.main' and e['consumer'] == 'dspark.blocks.0':
        removed.append({'old_graph': pin(graph_paths[0]), 'edge_index': j,
                        'edge': e, 'reason': 'Aux/context projection feeds shared KV. Query hidden starts at the draft embedding/repeat port; G5 DSPARK_BOUNDARIES already corrected this.'})
    else:
        edges.append(dict(copy.deepcopy(e), ancestry={'graph': 'G1-v88', 'edge_index': j}, accounting_level='macro'))
assert len(removed) == 1
for n in nodes:
    if n['id'] == 'engram_lookup':
        n['historical_kind'] = n['kind']; n['kind'] = 'ENGRAM_LOOKUP_PUBLIC_INTERFACE'
        n['selected_positive_path'] = 'NPU_HOST_UVA_GATHER_DEQUANT'
        n['baseline_async_overlap'] = False
        n['evidence'] = [str(H177/'loaded-module-instance-census.json'), str(H177/'operator-parent-scope-census.json')]
        n['alternative_paths'] = 'CPU/native/device/formats/collectives/events remain source-conditional; do not infer hardware traffic from table allocation.'

def node(id, kind, modules, **kw):
    assert id not in node_ids, id
    node_ids.add(id); nodes.append({'id': id, 'kind': kind, 'modules': modules, **kw})

def edge(a, b, value, kind='RAW', **kw):
    edges.append({'producer': a, 'consumer': b, 'value': value,
                  'dependency_kind': kind, 'iteration_offset': 0,
                  'source_evidence': 'public contract overlay; see port review and pinned source templates', **kw})

instances = []
def instance(id, template, parent, modules):
    t = templates[template]
    for n in t['nodes']:
        node(id + '/' + n['id'], n.get('kind', 'SOURCE_TEMPLATE_NODE'), modules,
             template=template, template_local_id=n['id'], parent_macro=parent,
             accounting_level='refinement', source_node=n)
    for j, e in enumerate(t['edges']):
        # Stateful updates are retained as effect edges; a cycle is not a same-iteration precedence proof.
        e = copy.deepcopy(e)
        e['producer'] = id+'/'+e['producer']; e['consumer'] = id+'/'+e['consumer']
        e.update(ancestry={'template': template, 'edge_index': j}, accounting_level='refinement')
        edges.append(e)
    instances.append({'id': id, 'template': template, 'parent_macro': parent,
                      'accounting': 'macro OR refinement, never both',
                      'native_lowering_or_instance_order_proved': False})

for r in roles:
    i = r['layer']; a = f'target.layers.{i}.attention'
    instance(a+'.detail', r['attention_template'], a, ['M07','M08','M09','M10','M11','M21','M23','M24'])
    edge(f'target.layers.{i}.input_norm', a+'.detail/x', 'rounded normalized hidden, with original CP/SP row restoration', 'PUBLIC_PORT_BINDING')
    edge(a+'.detail/reduce_scatter', f'target.layers.{i}.hc_attn_post', 'attention output, preserving selected shard/reduction/layout', 'PUBLIC_PORT_BINDING')
    instance(f'target.layers.{i}.moe.detail', 'MOE_WRAPPER', f'target.layers.{i}.moe', ['M12','M13','M23'])
    edge(f'target.layers.{i}.ffn_norm', f'target.layers.{i}.moe.detail/x', 'BF16 output of NormCast', 'PUBLIC_PORT_BINDING')
    edge(f'target.layers.{i}.ffn_norm', f'target.layers.{i}.moe.detail/x_fp32', 'FP32 widening of rounded BF16 output, not unrounded normalization', 'PUBLIC_PORT_BINDING')
    edge(f'target.layers.{i}.moe.detail/returned_moe_output', f'target.layers.{i}.hc_ffn_post', 'selected shared/routed result with required reductions', 'PUBLIC_PORT_BINDING')
    instance(f'target.layers.{i}.moe.mc2', 'MC2_INTERNAL_KNOWN_STAGES', f'target.layers.{i}.moe.detail/dispatch_ffn_combine', ['M12','M23'])

instance('draft.pipeline', 'DSPARK_BOUNDARIES', 'dspark.head', ['M15','M16','M18','M19','M20','M21','M24'])
instance('draft.shared_kv', 'DSPARK_SHARED_KV', 'dspark.main', ['M18','M21','M24'])
for i in [37,38,39]:
    edge(f'target.layers.{i}.aux', 'draft.pipeline/aux', 'ordered selected target auxiliary states, preserving original saved HC means', 'PUBLIC_PORT_BINDING')
edge('draft.pipeline/main_norm', 'draft.shared_kv/context', 'normalized context; distinct from draft token/query hidden', 'PUBLIC_PORT_BINDING')
for i in range(3):
    edge('draft.shared_kv/persistent_swa_state', f'draft.pipeline/block{i}', 'shared context KV with slots/positions/generation', 'PERSISTENT_STATE_READ')
    edge(f'draft.pipeline/block{i}', f'dspark.blocks.{i}', 'same semantic decoder block, alias of macro not an extra execution', 'ABSTRACTION_BINDING')
edge('draft.pipeline/draft_token_output', 'verify_accept', 'untrimmed proposal IDs and optional required probabilities; keep/trim handled separately', 'PUBLIC_PORT_BINDING')

# Exact per-layer value flow; cache source and index source are different namespaces.
cross_layer = []
for r in roles:
    i = r['layer']; a = f'target.layers.{i}.attention.detail'
    if r['index_source_layer'] is not None and not r['is_index_source']:
        s = r['index_source_layer']
        edge(f'target.layers.{s}.attention.detail/indices_store', a+'/indices_in',
             f'TopK value epoch from source layer {s}', 'LAYER_STATE_RAW')
        cross_layer.append({'kind':'index', 'source':s, 'consumer':i})
    if r['kv_source_layer'] is not None and not r['is_kv_source']:
        s = r['kv_source_layer']
        edge(f'target.layers.{s}.attention.detail/cmp_store', a+'/cache_in',
             f'compressed KV source {s}, valid request/position/page ranges', 'LAYER_STATE_RAW')
        cross_layer.append({'kind':'compressed_KV', 'source':s, 'consumer':i})
    if r['uses_candidate_filter'] and r['is_index_source']:
        edge('target.layers.20.attention', a+'/candidate_state_and_lengths',
             'preserved candidate-publication output of source layer20 plus original block/length mapping; not TopK value equality', 'CONDITIONAL_PUBLIC_STATE_PORT',
             condition='candidate state publication exists with the original selected-source schema; exact internal writer/representation is opaque')

# Versioned shared-bank effects. These express range retirement obligations, not full-kernel barriers.
bank = 'state.topk_shared_bank'
node(bank, 'VERSIONED_PERSISTENT_INDEX_BANK', ['M09','M10','M18','M21','M22','M24'],
     shape=[8192,512], dtype='INT32', allocated_bytes=16777216,
     namespace=['worker/pid','device','StorageImpl','allocation generation','valid row range','value epoch'],
     positive_alias_evidence=str(N/'shared-profile-analysis-v176/public-bank-epoch-and-role-review.json'),
     alias_scope='positive EP0 target/draft capture/formal identity only; no virtual-address cross-rank alias inference')
epochs = []
sources = cfg['index_source_layer_ids']
for j, s in enumerate(sources):
    readers = [r['layer'] for r in roles if r['index_source_layer'] == s]
    writer = f'target.layers.{s}.attention.detail/indices_store'
    next_writer = f'target.layers.{sources[j+1]}.attention.detail/indices_store' if j+1<len(sources) else 'state.draft_index_writer_or_preserved_bank'
    if next_writer.startswith('state.') and next_writer not in node_ids:
        node(next_writer, 'GUARDED_DRAFT_REUSE_OR_STATE_PRESERVATION', ['M18','M21','M24'],
             condition='only when that draft path actually writes overlapping shared-bank ranges')
    edge(writer, bank, f'publish source{s} epoch and valid ranges', 'BANK_WRITE_EFFECT')
    for l in readers:
        reader = f'target.layers.{l}.attention.detail/attention'
        edge(bank, reader, f'consume source{s} epoch for layer{l}', 'BANK_RANGE_RAW', epoch=f'target_index_source_{s}')
        edge(reader, next_writer, 'last read of overlapping range before overwrite; queued use must retire', 'BANK_RANGE_WAR',
             condition='same worker/device/allocation generation and overlapping physical row range; no required whole-operator barrier',
             range_identity='symbolic until actual views/valid rows bound')
    edge(writer, next_writer, 'ordered writes of overlapping range', 'BANK_RANGE_WAW', condition='same range and allocation generation')
    epochs.append({'source_layer':s,'reader_layers':readers,'next_possible_writer':next_writer,
                   'copy_on_write_full_live_range_bytes':'R_live * 512 * 4 (conditional whole-range versioning choice)',
                   'not_universal_required_allocation':True})

# The final target epoch also survives into the next iteration unless a guarded draft write intervenes.
next_target_writer = 'target.layers.2.attention.detail/indices_store'
for l in epochs[-1]['reader_layers']:
    edge(f'target.layers.{l}.attention.detail/attention', next_target_writer,
         'last target read before next-iteration overlapping overwrite', 'BANK_RANGE_WAR', iteration_offset=1,
         condition='same allocation generation and overlapping live range; original call/queue ordering retained')
edge('target.layers.36.attention.detail/indices_store', next_target_writer,
     'source36 epoch to next-iteration source2 epoch', 'BANK_RANGE_WAW', iteration_offset=1,
     condition='same allocation generation and overlapping physical range')
edge('state.draft_index_writer_or_preserved_bank', next_target_writer,
     'all actual draft/native readers of any intervening epoch must retire before reuse',
     'CONDITIONAL_BANK_EPOCH_RETIREMENT', iteration_offset=1,
     condition='intervening draft use exists; no unobserved native reader is assumed absent')

save('hierarchical-G0-public-dependency-overlay.json', {
    'status':'PUBLIC_OPERATOR_DEPENDENCY_REFINEMENT_AND_KNOWN_CORRECTIONS;NOT_NATIVE_RUNTIME_DAG',
    'pins':[pin(p) for p in graph_paths]+[pin(H/'layer-role-and-selector-bindings.json'),pin(H/'exact-guard-operator-contract-review.json')],
    'nodes':nodes,'edges':edges,'instances':instances,'cross_layer_bindings':cross_layer,
    'shared_topk_epochs':epochs,'removed_stale_macro_edges':removed,
    'abstraction_rules':[
        'Use macro or refinement accounting; ABSTRACTION_BINDING and PUBLIC_PORT_BINDING are semantic identities/interfaces, not extra kernels.',
        'All paths are conditional unions. Mutually exclusive template paths cannot be charged as simultaneous work.',
        'Persistent state effect edges can be cyclic across epochs; this is not a measured DAG or schedule.',
        'Native public composites preserve all original output/alias/state/communication effects. Private splitting requires extra evidence.',
        'Physical disjointness, deadness and readiness are never inferred from an unobserved node or missing trace record.'
    ],
    'remaining_public_graph_work':[
        'Instantiate host/request/metadata/replay/output and score-selection boundary ports from existing semantic rows.',
        'Audit exact all-consumer/alias/state envelopes of remaining templates; opaque native effects stay explicit.',
        'Check all42 family footprints and all26 module ports against the corrected hierarchy, then run the declared rule round.'
    ],
    'whole_scope_complete':False})

unknown_endpoints = sorted({e[k] for e in edges for k in ['producer','consumer'] if e[k] not in node_ids})
assert not unknown_endpoints, unknown_endpoints
assert len(epochs)==8
assert sum(len(e['reader_layers']) for e in epochs)==38
assert all(r['index_source_layer']==24 and r['kv_source_layer']==20 for r in roles[24:28])
checks = {'status':'PASS','source_hashes_match':True,'target_layer_roles':40,
          'hierarchical_nodes':len(nodes),'semantic_and_effect_edges':len(edges),
          'template_instances':len(instances),'cross_layer_bindings':len(cross_layer),
          'unresolved_endpoint_ids':unknown_endpoints,'shared_index_epochs':len(epochs),
          'index_reader_layers':38,'stale_DSpark_direct_context_to_decoder_edges_removed':len(removed),
          'whole_scope_complete':False,'new_NPU_or_model_work':False,'target_source_mutation':False}
save('public-dependency-overlay-checks.json',checks)
print(json.dumps(checks,ensure_ascii=False))
