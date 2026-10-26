#!/usr/bin/env python3
"""Run507 reversible source patcher. Check --source-dir compiles pinned copies only."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
ASC=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
VLLM=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm')
SOURCES=dict(runtime=ROOT/'runtime/extreme_decode.py',serving=ROOT/'runtime/fixed_serving.py',
             target=ROOT/'runtime/target_adapter.py',handoff=ROOT/'bootstrap/vllm_target_handoff.py',
             graph=ASC/'compilation/acl_graph.py',
             comm=VLLM/'distributed/device_communicators/base_device_communicator.py')
PINS=dict(runtime='eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499',
serving='137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a',
target='c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5',
handoff='2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5',
graph='6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6',
comm='c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221')
MARK='# EXTREME_RUN507_FRONTIER'

def sha(x):return hashlib.sha256(x).hexdigest()
def insert(s,anchor,block,before=False):
    if s.count(anchor)!=1:raise ValueError(f'Run507 anchor count {s.count(anchor)}: {anchor!r}')
    return s.replace(anchor,(block+anchor) if before else (anchor+block))

def patch(key,s):
    if MARK in s:raise ValueError('already patched')
    if key=='runtime':
        s=insert(s,'        self._profile_cycle_begin()\n','''        # EXTREME_RUN507_FRONTIER
        from scripts import loop079_target_frontier_run507 as _f478
        _f478.step(self)
''',True)
        return insert(s,'            if diag is not None:\n                diag["target_argmax"]','''            # EXTREME_RUN507_FRONTIER
            _f478.finish_step()
''',True)
    if key=='serving':
        s=insert(s,'        cycles = 0\n','''        # EXTREME_RUN507_FRONTIER
        from scripts import loop079_target_frontier_run507 as _f478
        _f478.start(self)
''',True)
        return insert(s,'        counts_cpu = count_history[:cycles].cpu()\n','''        # EXTREME_RUN507_FRONTIER: after ordinary cohort drain
        _f478.export(self, counts_cpu, cycles)
''')
    if key=='target':
        anchor='''        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
'''
        replacement='''        # EXTREME_RUN507_FRONTIER
        from scripts import loop079_target_frontier_run507 as _f478
        with _f478.target_scope(state):
            _f478.target_begin(state)
            model_output = self.binding.forward(
                state.target_input_ids,
                state.target_positions,
            )
'''
        if s.count(anchor)!=1:raise ValueError('Target forward anchor')
        s=s.replace(anchor,replacement)
        return insert(s,'        sample_hidden = hidden_states[state.target_logits_indices]\n','''        # EXTREME_RUN507_FRONTIER
        _f478.target_after(hidden_states, sample_hidden, state)
''')
    if key=='handoff':
        s=insert(s,'def _gather_hidden(hidden: torch.Tensor) -> torch.Tensor:\n    context = get_forward_context()\n','''    # EXTREME_RUN507_FRONTIER
    from scripts import loop079_target_frontier_run507 as _f478
    _f478.hidden_begin(hidden)
''',before=False)
        s=insert(s,'    return hidden\n','''    # EXTREME_RUN507_FRONTIER
    _f478.hidden_end(hidden)
''',True)
        if s.count('''            if self.graph_update is not None and self.graph_update_before:
                self.graph_update(context, num_tokens)
''') != 1:raise ValueError('graph update before anchor')
        if s.count('''            if self.graph_update is not None and not self.graph_update_before:
                self.graph_update(context, num_tokens)
''') != 1:raise ValueError('graph update after anchor')
        s=s.replace('''            if self.graph_update is not None and self.graph_update_before:
                self.graph_update(context, num_tokens)
''','''            if self.graph_update is not None and self.graph_update_before:
                # EXTREME_RUN507_FRONTIER
                from scripts import loop079_target_frontier_run507 as _f478
                _f478.update_begin(self, "before")
                self.graph_update(context, num_tokens)
                _f478.update_end()
''')
        s=s.replace('''            if self.graph_update is not None and not self.graph_update_before:
                self.graph_update(context, num_tokens)
''','''            if self.graph_update is not None and not self.graph_update_before:
                # EXTREME_RUN507_FRONTIER
                from scripts import loop079_target_frontier_run507 as _f478
                _f478.update_begin(self, "after")
                self.graph_update(context, num_tokens)
                _f478.update_end()
''')
        s=insert(s,'            if context.flash_comm_v1_enabled:\n                output = _gather_output(output)\n','''            # EXTREME_RUN507_FRONTIER
            from scripts import loop079_target_frontier_run507 as _f478
            _f478.gather_begin(output, context, self)
''',True)
        return insert(s,'            if context.flash_comm_v1_enabled:\n                output = _gather_output(output)\n','''            # EXTREME_RUN507_FRONTIER
            _f478.gather_end(output)
''')
    if key=='graph':
        original='''    impl_cls = attn_backend.get_impl_cls()
    impl_cls.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )
'''
        observed='''    impl_cls = attn_backend.get_impl_cls()
    # EXTREME_RUN507_FRONTIER: observe the existing resolved selected update call
    update_callable = impl_cls.update_graph_params
    from scripts import loop079_target_frontier_run507 as _f478
    _f478.actual_update_begin(attn_backend, impl_cls, update_callable, update_stream,
                              forward_context, num_tokens, draft_attn_metadatas)
    update_callable(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )
    _f478.actual_update_end()
'''
        if s.count(original)!=1:raise ValueError('FULL update implementation anchor')
        s=s.replace(original,observed)
        s=insert(s,'        forward_context = get_forward_context()\n        batch_descriptor = forward_context.batch_descriptor\n','''        # EXTREME_RUN507_FRONTIER: bound on every replay branch
        from scripts import loop079_target_frontier_run507 as _f478
''')
        s=insert(s,'        if entry.aclgraph is None:\n','''            # EXTREME_RUN507_FRONTIER: fail selected cache misses before capture
            _f478.cache_miss(self, entry, forward_context)
''')
        s=insert(s,'            entry.aclgraph = aclgraph\n','''            # EXTREME_RUN507_FRONTIER: Host-only capture registration
            _f478.capture(self, entry, aclgraph, output, args, kwargs, forward_context, _EXTRA_CTX.is_draft_model)
''')
        anchor='''        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()
'''
        replacement='''        if not self.enable_enpu and need_sync:
            # EXTREME_RUN507_FRONTIER
            _f478.sync_begin(self, entry, is_draft_eagle, need_sync)
            torch.npu.current_stream().synchronize()
            _f478.sync_end()
'''
        if s.count(anchor)!=1:raise ValueError('FULL sync anchor')
        s=s.replace(anchor,replacement)
        s=insert(s,'        entry.aclgraph.replay()\n','''        # EXTREME_RUN507_FRONTIER
        _f478.replay_pre(self, entry, args, kwargs, is_draft_eagle, need_sync)
''',True)
        return insert(s,'        entry.aclgraph.replay()\n','''        # EXTREME_RUN507_FRONTIER
        _f478.replay_post(entry)
''')
    s=insert(s,'        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)\n','''        # EXTREME_RUN507_FRONTIER
        from scripts import loop079_target_frontier_run507 as _f478
        _f478.native_pre(self, output_tensor, input_)
''',True)
    return insert(s,'        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)\n','''        # EXTREME_RUN507_FRONTIER
        _f478.native_post(self, output_tensor, input_)
''')

def atomic(path,data):
    fd,tmp=tempfile.mkstemp(prefix='.run494-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,path.stat().st_mode & 0o777);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','install','restore'))
    p.add_argument('--source-dir',type=Path,help='check only: pinned copied originals, never live sources')
    p.add_argument('--state-dir',type=Path)
    p.add_argument('--record',type=Path,required=True)
    a=p.parse_args(); rows=[]
    if a.action in ('check','install'):
        if a.source_dir is not None and a.action!='check':raise ValueError('source-dir only for check')
        prepared=[]
        for key,path in SOURCES.items():
            source=(a.source_dir/f'{key}.py') if a.source_dir else path
            old=source.read_bytes()
            if sha(old)!=PINS[key]:raise ValueError(f'Run507 source pin mismatch: {source}')
            new=patch(key,old.decode()).encode();compile(new,str(path),'exec')
            prepared.append((key,path,old,new))
            rows.append(dict(key=key,source=str(path),original=sha(old),patched=sha(new)))
        helper=ROOT/'scripts/loop079_target_frontier_run507.py'
        compile(helper.read_bytes(),str(helper),'exec')
        if a.action=='install':
            if a.state_dir is None or a.state_dir.exists():raise ValueError('fresh state-dir required')
            a.state_dir.mkdir(parents=True)
            for key,_,old,_ in prepared:(a.state_dir/f'{key}.orig').write_bytes(old)
            (a.state_dir/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
            try:
                for _,path,old,new in prepared:
                    if path.read_bytes()!=old:raise ValueError(f'source changed: {path}')
                    atomic(path,new)
            except BaseException:
                for _,path,old,new in prepared:
                    if path.read_bytes()==new:atomic(path,old)
                raise
    else:
        if a.source_dir is not None or a.state_dir is None:raise ValueError('restore state-dir required')
        rows=json.loads((a.state_dir/'manifest.json').read_text())
        if [r['key'] for r in rows]!=list(SOURCES):raise ValueError('manifest keys')
        for row in rows:
            path=SOURCES[row['key']];backup=(a.state_dir/f"{row['key']}.orig").read_bytes()
            if str(path)!=row['source'] or sha(path.read_bytes())!=row['patched'] or sha(backup)!=row['original']:
                raise ValueError(f'refuse changed source/backup: {path}')
        for row in rows:atomic(SOURCES[row['key']],(a.state_dir/f"{row['key']}.orig").read_bytes())
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(dict(action=a.action,fixture=str(a.source_dir) if a.source_dir else None,files=rows),indent=2)+'\n')
    print(json.dumps(dict(action=a.action,files=len(rows))))
if __name__=='__main__':main()
