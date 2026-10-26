"""Host-only spans on saved Run678 candidate; off-path, no new device operations."""
import ast
from loop081_ordinary_boundary_patch_run676 import one, runner as old_runner

def runner(src):
    src=old_runner(src).replace('diagnostics.ordinary_boundary.observer','diagnostics.startup_boundary.observer').replace('EXTREME_ORDINARY_BOUNDARY_DIR','EXTREME_STARTUP_BOUNDARY_DIR')
    anchor="            if _ob_active:_ob_event(self,'runtime_built',submit_id=self._ob_current_submit,req_ids=list(_extreme_req_ids))"
    src=one(src,anchor,anchor+"\n            if _ob_active:_extreme_runtime.proposer._startup_boundary_owner = self")
    anchor='                _extreme_runtime.target_metadata.capture_graph(_extreme_runtime.state)'
    src=one(src,anchor,"                if _ob_active:_ob_event(self,'capture_begin',submit_id=self._ob_current_submit)\n"+anchor+"\n                if _ob_active:_ob_event(self,'capture_end',submit_id=self._ob_current_submit)")
    block="                if _ob_active:\n                    from diagnostics.startup_boundary.observer import flush as _ob_flush\n                    _ob_flush(self,_rank,_cohort_index)\n"
    src=one(src,block,'')
    anchor='                    extreme_bulk_output=True,\n                )\n                return None'
    src=one(src,anchor,"""                    extreme_bulk_output=True,
                )
                if _ob_active:
                    _ob_event(self,'output_ready',submit_id=self._ob_current_submit,req_ids=list(_extreme_req_ids))
                    from diagnostics.startup_boundary.observer import flush as _ob_flush
                    _ob_flush(self,_rank,_cohort_index)
                return None""")
    compile(src,'run679_runner','exec')
    return src

def handoff(src):
    tree=ast.parse(src)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DirectDSparkHandoff')
    events={'certify_startup_slots':'certificate','_flush_slot_refresh_audit':'audit_flush','validate_host_mirrors':'validation'}
    lines=src.splitlines(keepends=True)
    funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in events]
    assert len(funcs)==len(events)
    for n in sorted(funcs,key=lambda n:n.lineno,reverse=True):
        start=n.body[0].lineno-1;end=n.end_lineno
        body=lines[start:end]
        assert all(not x.strip() or x.startswith('        ') for x in body)
        event=events[n.name]
        lines[start:end]=[f"        self._startup_boundary_mark('{event}_begin')\n",'        try:\n']+['    '+x if x.strip() else x for x in body]+['        finally:\n',f"            self._startup_boundary_mark('{event}_end')\n"]
    src=''.join(lines)
    anchor='    def certify_startup_slots('
    assert src.count(anchor)==1
    helper="""    def _startup_boundary_mark(self, kind):
        owner = getattr(self, '_startup_boundary_owner', None)
        if owner is not None:
            from diagnostics.startup_boundary.observer import worker_event
            worker_event(owner, kind, submit_id=owner._ob_current_submit)

"""
    src=src.replace(anchor,helper+anchor,1)
    compile(src,'run679_handoff','exec')
    return src
