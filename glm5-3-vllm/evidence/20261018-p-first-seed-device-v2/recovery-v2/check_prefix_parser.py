from pathlib import Path
import copy,importlib.util,json
r=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('prefix_parser',r/'summarize_prefix_paths.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=[]
for rank in range(16):
 for requested in [False,True]:
  records.append({'rank':rank,'phase':'execute_model','requested':requested,'request_ids':['real'],
   'graphs':[{'tokens':6,'requested':requested,'selected':requested,'native_capture':{'calls':75,'selected_calls':0,'output_list_types':[1]},'prefix_capture':{'calls':75,'selected_calls':75,'output_list_types':[0]}}],
   'dispatch':[{'N':6,'prefix_requested':requested,'selected':requested,'output_list_type':0 if requested else 1}]*2})
a=m.summarize_prefix_paths(records);assert a['all16_paths_verified'] and all(z['selected_graph_tokens']==[6] and z['selected_dispatch_N']==[6] for z in a['per_rank'])
mutations=[lambda r:r.pop(),lambda r:r[0]['dispatch'][0].update(output_list_type=0),lambda r:r[0]['dispatch'][0].update(selected=True),lambda r:r[1]['graphs'][0]['prefix_capture'].update(calls=74),lambda r:r[1]['graphs'][0].update(selected=False),lambda r:r[1]['dispatch'][0].update(selected=False),lambda r:r[1]['graphs'][0].update(requested=False),lambda r:r[1].update(rank=True),lambda r:r[0].update(dispatch=None),lambda r:r[1]['graphs'][0]['prefix_capture'].update(output_list_types=[False]),lambda r:r[0].update(request_ids='bad'),lambda r:r[0].update(phase='bad')]
# Remove every record of one rank for actual missing-rank case.
mutations[0]=lambda r:r.__delitem__(slice(-2,None))
for change in mutations:
 bad=copy.deepcopy(records);change(bad);assert not m.summarize_prefix_paths(bad)['all16_paths_verified']
assert not m.summarize_prefix_paths([])['all16_paths_verified']
p=r/'zcode-prefix-paths-v1/review.json';d=json.loads(p.read_text());d['tests']={'valid_all16':True,'rejected_mutations':len(mutations),'empty_rejected':True};p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
