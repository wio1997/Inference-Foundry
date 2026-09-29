"""Reuse Run666 Graph arithmetic unchanged; choose mode at cohort construction."""
import hashlib,json
from pathlib import Path
import loop081_metadata_graph_patch_run666 as old
root=Path('/data/wio/Inference_Foundry');out=root/'evidence/20260929_loop081_bound/run675'
cand=out/'candidate';cand.mkdir(parents=True,exist_ok=False)
pin={'sources':{},'helpers':{}}
for key,(path,expected) in old.SOURCES.items():
 raw=path.read_bytes();assert old.sha(raw)==expected
 new=old.PATCH[key](raw.decode())
 if key=='metadata':
  new=old.one(new,'        self._graph_enabled = os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1"','''        from serving.cohort_mode import read_cohort_mode
        self._graph_mode = read_cohort_mode(os.environ["EXTREME_METADATA_MODE_FILE"])
        self._graph_enabled = self._graph_mode["enabled"]''')
 else:
  new=old.one(new,'            if os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1":','''            if (_extreme_runtime.target_metadata is not None
                    and _extreme_runtime.target_metadata._graph_enabled):''')
  new=old.one(new,'                    "metadata_graph_capture": bool(','''                    "metadata_graph_mode": dict(_extreme_runtime.target_metadata._graph_mode),
                    "metadata_graph_capture": bool(''')
 compile(new,str(path),'exec');(cand/(key+'.py')).write_text(new)
 pin['sources'][key]={'path':str(path),'before_sha256':old.sha(raw),'candidate_sha256':old.sha(new.encode())}
for path in (root/'serving/cohort_mode.py',Path(old.__file__),Path(__file__).resolve()):pin['helpers'][str(path)]=old.sha(path.read_bytes())
(out/'candidate_manifest.json').write_text(json.dumps(pin,indent=2)+'\n');print(json.dumps(pin,indent=2))
