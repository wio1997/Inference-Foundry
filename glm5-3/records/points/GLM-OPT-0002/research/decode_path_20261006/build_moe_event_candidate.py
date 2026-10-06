"""Build H5 from exact installed byte identities, into a new directory only."""
import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path


def build(original, dest, identities):
    assert not dest.exists()
    dest.mkdir(parents=True)
    diff=[];result={}
    for name,identity in identities.items():
        raw=(original/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==identity['sha256']
        newline='\r\n' if b'\r\n' in raw else '\n'
        old=raw.decode().replace('\r\n','\n');new=old
        if name=='utils.py':
            marker='def create_hccl_pg_options(group_name: str):'
            assert new.count(marker)==1
            helper='''def maybe_record_moe_event() -> torch.npu.Event | None:
    """Record a shared-expert dependency only when it crosses streams."""
    if not get_ascend_config().multistream_overlap_shared_expert:
        return None
    return torch.npu.current_stream().record_event()


'''
            new=new.replace(marker,helper+marker)
        elif name.endswith('/shared_experts.py'):
            assert new.count('before_routed_experts: torch.npu.Event')==1
            new=new.replace('before_routed_experts: torch.npu.Event','before_routed_experts: torch.npu.Event | None')
            old_wait='torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)'
            assert new.count(old_wait)==3
            new=new.replace(old_wait,'maybe_wait_event(fused_moe_evts.before_routed_experts)')
            marker='        mode = self.parallel_mode()\n'
            assert new.count(marker)==1
            new=new.replace(marker,'        if self.multistream_overlap:\n            assert fused_moe_evts.before_routed_experts is not None, "Missing cross-stream shared-expert input event"\n'+marker)
        else:
            expected={'fused_moe.py':6,'moe_comm_method.py':2,'moe_mlp.py':3}[Path(name).name]
            # Only the named shared-expert producer variables are changed;
            # after_routed_finalize retains its original guarded record.
            lines=[];changed=0
            for line in new.splitlines(keepends=True):
                if any(token in line for token in ['before_routed_experts =','after_routed_experts =','before_dispatch_evt =','before_combine_evt =','before_gmm2_evt =']) and 'torch.npu.current_stream().record_event()' in line:
                    line=line.replace('torch.npu.current_stream().record_event()','maybe_record_moe_event()');changed+=1
                lines.append(line)
            assert changed==expected,(name,changed)
            new=''.join(lines)
            # Extend the existing import at its original position. No new
            # runtime module and no earlier import/circularity change.
            imports=[n for n in ast.parse(new).body if isinstance(n,ast.ImportFrom) and n.module=='vllm_ascend.utils']
            lines=new.splitlines(keepends=True)
            if not imports:
                assert name.endswith('/moe_comm_method.py')
                end=max(n.end_lineno for n in ast.parse(new).body if isinstance(n,(ast.Import,ast.ImportFrom)))
                lines.insert(end,'from vllm_ascend.utils import maybe_record_moe_event\n')
            else:
                assert len(imports)==1
                node=imports[0];text=''.join(lines[node.lineno-1:node.end_lineno])
                if '(' in text:
                    assert text.endswith(')\n')
                    text=text[:-2]+'    maybe_record_moe_event,\n)\n'
                else:
                    text=text.rstrip('\n')+', maybe_record_moe_event\n'
                lines[node.lineno-1:node.end_lineno]=[text]
            new=''.join(lines)
        ast.parse(new)
        candidate=new.replace('\n',newline).encode()
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(candidate)
        result[name]=dict(original_sha256=identity['sha256'],candidate_sha256=hashlib.sha256(candidate).hexdigest(),bytes=len(candidate))
        diff.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/vllm_ascend/'+name,tofile='b/vllm_ascend/'+name))
    return ''.join(diff),result


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    original=Path(sys.argv[1]) if len(sys.argv)>1 else here/'event_originals'
    dest=Path(sys.argv[2]) if len(sys.argv)>2 else here/'event_candidate'
    diff,identity=build(original,dest,json.loads((here/'event_source_identity.json').read_text()))
    (here/'moe_event.patch').write_text(diff)
    (here/'moe_event_patch_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity,indent=2))
