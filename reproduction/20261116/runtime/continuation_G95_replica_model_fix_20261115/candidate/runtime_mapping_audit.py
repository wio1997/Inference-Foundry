"""Opt-in, fail-closed audit for this frozen G95 E384/EP16/TP1 W8A8 load.

Root calls install() before any FusedMoE construction, finalize_target() after
target loader join and finalize_draft() after draft load. No env => no-op.
No loader math is replaced. This module deliberately rejects other geometry.
"""
from __future__ import annotations
import hashlib, inspect, json, os, threading, time
from pathlib import Path

_audit = None
ENV = 'SGLANG_G95_MAPPING_AUDIT_PLAN'
FIELDS = ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale',
          'w13_weight_offset', 'w2_weight_offset')

def require(value, message):
    if not value:
        raise RuntimeError('G95 mapping audit: ' + message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Audit:
    def __init__(self, plan):
        import torch
        self.torch = torch
        self.plan = json.loads(Path(plan).read_text())
        require(set(self.plan) == {'mapping_path','mapping_sha256','output_dir','source_sha256'}, 'plan keys')
        require(sha(self.plan['mapping_path']) == self.plan['mapping_sha256'], 'mapping SHA')
        data = json.loads(Path(self.plan['mapping_path']).read_text())
        require(set(data) == {'physical_to_logical_map'}, 'mapping schema')
        self.rows = data['physical_to_logical_map']
        require(len(self.rows) == 40, 'layer count')
        for row in self.rows:
            require(len(row) == 400 and all(type(x) is int for x in row)
                    and set(row) == set(range(384)), 'logical coverage / physical400')
        self.inverse = [{logical: [p for p,x in enumerate(row) if x == logical] for logical in range(384)} for row in self.rows]
        self.lock = threading.Lock()
        self.compare_lock = threading.Lock()  # <=8MiB scratch across all loader threads
        self.local = threading.local()
        self.layers = {}; self.drafts = {}; self.started = set(); self.done = set()
        self.byte_count = 0; self.rank = None; self.started_at = time.monotonic()
        self.finalized = False

    def source(self, module, key):
        require(sha(inspect.getsourcefile(module)) == self.plan['source_sha256'][key], 'source SHA '+key)

    def created(self, obj):
        draft = bool(obj.is_draft)
        require(bool(obj.uses_expert_location) == (not draft), 'target/draft uses_expert_location')
        if draft:
            with self.lock:
                self.drafts[id(obj)] = {'layer_id':int(obj.layer_id), 'loads':0, 'uses_expert_location':False}
            return
        require(obj.moe_ep_size == 16 and obj.moe_tp_size == 1 and obj.moe_tp_rank == 0, 'EP16/MoETP1')
        require(obj._num_global_routed == 400 and obj._num_local_routed == 25
                and obj.num_local_experts == 25 and not obj._has_fused_shared, 'routed25/no fused shared')
        require(obj.hidden_size == 5120 and obj.intermediate_size_per_partition == 2304, 'model dimensions')
        require(obj.moe_runner_config.is_gated, 'gated W13')
        require(not obj.use_padded_loading and not obj.use_presharded_weights, 'unexpected padded/presharded loader')
        require(not getattr(obj.quant_method,'load_up_proj_weight_first',False), 'unexpected W13 half order')
        require(type(obj.quant_method).__name__ == 'ModelSlimFusedMoEMethod', 'expected ModelSlim method')
        require(obj._expert_storage_rank == obj.moe_ep_rank, 'EP/storage rank')
        layer = int(obj.layer_id); rank = int(obj.moe_ep_rank)
        require(0 <= layer < 40 and 0 <= rank < 16, 'layer/rank range')
        params = {}
        for name in FIELDS:
            p = getattr(obj,name)
            require(not getattr(p,'_sglang_require_global_experts',False), name+' unexpectedly global')
            require(not getattr(p,'is_transposed',False), name+' unexpectedly transposed')
            dim = 4608 if name.startswith('w13') else 5120
            shape = (25,dim,5120 if name.startswith('w13') else 2304) if name.endswith('_weight') else (25,dim,1)
            require(tuple(p.shape) == shape, name+' shape '+str(tuple(p.shape)))
            require(p.dtype == (self.torch.int8 if name.endswith('_weight') else self.torch.float32), name+' dtype')
            params[id(p)] = name
        with self.lock:
            require(layer not in self.layers, 'duplicate target layer')
            require(self.rank in (None,rank), 'mixed EP ranks in process')
            self.rank = rank
            self.layers[layer] = {'object':obj, 'params':params}

    def public(self, original, obj, param, loaded_weight, weight_name, shard_id, expert_id):
        if obj.is_draft:
            require(not obj.uses_expert_location, 'draft mapped')
            with self.lock:
                self.drafts[id(obj)]['loads'] += 1
            return original(obj,param,loaded_weight,weight_name,shard_id,expert_id)
        require(not self.finalized, 'target weight load after finalized')
        require(getattr(self.local,'ctx',None) is None, 'nested public loader')
        require(type(expert_id) is int and 0 <= expert_id < 384, 'logical expert')
        self.local.ctx = (id(obj),id(param),expert_id,shard_id,weight_name)
        self.local.writes = 0
        try:
            result = original(obj,param,loaded_weight,weight_name,shard_id,expert_id)
            physical = self.inverse[obj.layer_id][expert_id]
            expected = sum(p // 25 == self.rank for p in physical)
            require(self.local.writes == expected, 'public loader local write count')
            return result
        finally:
            self.local.ctx = None

    def impl(self, original, obj, param, loaded_weight, weight_name, shard_id, expert_id):
        if obj.is_draft:
            return original(obj,param,loaded_weight,weight_name,shard_id,expert_id)
        ctx = getattr(self.local,'ctx',None)
        require(ctx is not None and ctx[:2] == (id(obj),id(param)), 'impl bypassed public logical context')
        logical = ctx[2]
        require(ctx[3:] == (shard_id,weight_name), 'public/impl arguments changed')
        physical = self.inverse[obj.layer_id][logical]
        require(any(p//25 == self.rank and expert_id == p%25 for p in physical), 'actual physical/local destination')
        name = self.layers[obj.layer_id]['params'].get(id(param))
        require(name in FIELDS and weight_name.endswith('.'+name), 'actual parameter identity/name')
        require(shard_id in ('w1','w3') if name.startswith('w13') else shard_id == 'w2', 'shard family')
        require(not getattr(param,'_sglang_require_global_experts',False), 'global flag changed')
        target = param.data[expert_id]
        if shard_id in ('w1','w3'):
            target = target.narrow(0,0 if shard_id == 'w1' else 2304,2304)
        require(tuple(target.shape) == tuple(loaded_weight.shape), 'checkpoint/destination shape')
        require(target.dtype == loaded_weight.dtype and target.is_contiguous(), 'dtype/contiguous contract')
        key = (int(obj.layer_id),int(expert_id),shard_id,name)
        with self.lock:
            require(key not in self.started, 'duplicate destination write '+str(key))
            self.started.add(key)
        result = original(obj,param,loaded_weight,weight_name,shard_id,expert_id)
        # No reference-loader replay and no second checkpoint read. Same source
        # tensor remains live from public call; strict TP1 shapes need no math.
        require(loaded_weight.is_contiguous(), 'noncontiguous checkpoint tensor')
        actual = target.view(-1); source = loaded_weight.view(-1)
        chunk = max(1,(8*1024*1024)//source.element_size())
        with self.compare_lock:
            for start in range(0,source.numel(),chunk):
                expected = source[start:start+chunk].to(device=actual.device, non_blocking=False)
                require(self.torch.equal(actual[start:start+chunk],expected), 'all-content mismatch '+str(key))
                del expected
        with self.lock:
            self.done.add(key); self.byte_count += source.numel()*source.element_size()
        self.local.writes += 1
        return result

    def finish(self):
        require(not self.finalized, 'finalize twice')
        require(set(self.layers) == set(range(40)), 'missing target layers')
        expected = {(layer,slot,shard,name) for layer in range(40) for slot in range(25)
                    for shard in ('w1','w3','w2') for suffix in ('weight','weight_scale','weight_offset')
                    for name in [('w2_' if shard == 'w2' else 'w13_')+suffix]}
        require(self.started == expected and self.done == expected, 'missing/inflight coverage')
        import torch.distributed as dist
        require(dist.is_initialized() and dist.get_world_size() == 16, 'actual world16')
        # Actual installed native dynamic lookup; all logical IDs on all layers.
        # Changed-input NPUGraph replay covers the new randint/indexing path.
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo, topk_ids_logical_to_physical
        metadata = get_global_expert_location_metadata()
        require(metadata.physical_to_logical_map_cpu.tolist() == self.rows, 'actual metadata/map mismatch')
        device=metadata.physical_to_logical_map.device
        ids=self.torch.arange(384,device=device,dtype=self.torch.int32).repeat(4).reshape(256,6)
        graph_layers=[];eligible_replicated_logicals=0
        def check(out,layer):
            require(out.dtype==ids.dtype and tuple(out.shape)==(256,6), 'route lookup shape/dtype')
            require(bool(((out>=0)&(out<400)).all().item()), 'physical range')
            require(self.torch.equal(metadata.physical_to_logical_map[layer][out.long()],ids.long()), 'actual dynamic route inverse')
        for layer in range(40):
            info=ExpertLocationDispatchInfo.init_new(layer)
            require(info.ep_dispatch_algorithm=='dynamic' and info.num_physical_experts==400 and not info.rank_invariant, 'actual dispatch mode')
            counts=info.partial_logical_to_all_physical_map_num_valid
            eligible_replicated_logicals+=int((counts>1).sum().item())
            for _ in range(3):check(topk_ids_logical_to_physical(ids,info),layer)
            self.torch.npu.synchronize()
            graph=self.torch.npu.NPUGraph()
            with self.torch.npu.graph(graph):out=topk_ids_logical_to_physical(ids,info)
            graph.replay();self.torch.npu.synchronize();check(out,layer)
            for shift in (17,137):
                ids.copy_((self.torch.arange(384,device=device,dtype=self.torch.int32).repeat(4).reshape(256,6)+shift)%384)
                graph.replay();self.torch.npu.synchronize();check(out,layer)
            graph_layers.append(layer);del graph,out
            require(ExpertLocationDispatchInfo.init_new(layer,is_draft=True) is None,'draft dispatch map')
        self.torch.npu.synchronize();self.torch.npu.empty_cache()
        self.finalized = True
        record = {'schema':1,'status':'ACTUAL_PREPROCESS_LOAD_CONTENT_AND_DYNAMIC_GRAPH_LOOKUP_PASS',
                  'mapping_sha256':self.plan['mapping_sha256'],'source_sha256':self.plan['source_sha256'],
                  'global_rank':dist.get_rank(),'ep_rank':self.rank,'pid':os.getpid(),'ep_size':16,'moe_tp_size':1,
                  'target_layers':40,'routed_slots_per_rank':25,'allcontent_eq_count':len(self.done),
                  'allcontent_bytes_compared':self.byte_count,'coverage_per_layer':225,
                  'drafts':list(self.drafts.values()),'dynamic_lookup_layers':40,'changed_input_graph_layers':graph_layers,'eligible_replicated_logicals':eligible_replicated_logicals,
                  'elapsed_seconds':time.monotonic()-self.started_at,
                  'scope':'All checkpoint destination bytes before ModelSlim postprocess; actual dynamic lookup synthetic all-ID inverse; draft loader/map exclusion. No postprocess, real router score/weight, full-model quality or performance acceptance.'}
        outdir = Path(self.plan['output_dir']); outdir.mkdir(parents=True,exist_ok=True)
        dest = outdir/('mapping_audit_rank%d.json'%dist.get_rank())
        require(not dest.exists(), 'receipt already exists')
        tmp = outdir/(dest.name+'.pid%d.tmp'%os.getpid())
        with tmp.open('x') as f: json.dump(record,f,indent=2); f.write('\n')
        os.replace(tmp,dest)
        return record

    def finish_draft(self):
        require(self.finalized, 'draft completion before target audit')
        require(self.drafts and all(d['loads'] > 0 for d in self.drafts.values()), 'draft exclusion not observed during actual load')
        from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
        for d in self.drafts.values():
            require(ExpertLocationDispatchInfo.init_new(d['layer_id'],is_draft=True) is None, 'draft dispatch not excluded')
        import torch.distributed as dist
        record={'schema':1,'status':'ACTUAL_DRAFT_LOAD_MAPPING_EXCLUSION_PASS',
                'global_rank':dist.get_rank(),'ep_rank':self.rank,'pid':os.getpid(),
                'source_sha256':self.plan['source_sha256'],
                'mapping_sha256':self.plan['mapping_sha256'],'drafts':list(self.drafts.values()),
                'scope':'Observed draft FusedMoE construction and actual loader calls exclude target mapping; draft numerical/state acceptance remains separate.'}
        dest=Path(self.plan['output_dir'])/('mapping_draft_audit_rank%d.json'%dist.get_rank())
        with dest.open('x') as f: json.dump(record,f,indent=2); f.write('\n')
        return record

def install():
    global _audit
    path = os.environ.get(ENV)
    if not path:
        return False
    require(_audit is None, 'install twice')
    from sglang.srt.layers.moe.fused_moe_triton import layer as mod
    import sglang.srt.eplb.expert_location_dispatch as dispatch
    import sglang.srt.eplb.expert_location as location
    a = Audit(path)
    require(set(a.plan['source_sha256']) == {'layer','dispatch','location'}, 'source hash keys')
    for module,key in ((mod,'layer'),(dispatch,'dispatch'),(location,'location')): a.source(module,key)
    cls = mod.FusedMoE
    init,public,impl = cls.__init__,cls.weight_loader,cls._weight_loader_impl
    def init_hook(obj,*args,**kwargs):
        init(obj,*args,**kwargs); a.created(obj)
    def public_hook(obj,param,loaded_weight,weight_name,shard_id,expert_id):
        return a.public(public,obj,param,loaded_weight,weight_name,shard_id,expert_id)
    def impl_hook(obj,param,loaded_weight,weight_name,shard_id,expert_id):
        return a.impl(impl,obj,param,loaded_weight,weight_name,shard_id,expert_id)
    cls.__init__,cls.weight_loader,cls._weight_loader_impl = init_hook,public_hook,impl_hook
    _audit = a
    return True

def finalize_target():
    if not os.environ.get(ENV):
        return None
    require(_audit is not None, 'enabled but not installed')
    return _audit.finish()

def finalize_draft():
    if not os.environ.get(ENV):
        return None
    require(_audit is not None, 'enabled but not installed')
    return _audit.finish_draft()
