#!/usr/bin/env python3
"""Read-only metadata audit; no prompts, full outputs, auth config or model I/O exported."""
import json,pathlib,sqlite3
R=pathlib.Path(__file__).resolve().parents[1];db=sqlite3.connect('file:/Users/wio/.zcode/cli/db/db.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row;aud=[]
for n in range(1,10):
 tag=f'L031-R{n:03}';sid=json.loads((R/f'runs/{tag}-zcode/stdout.jsonl').read_text())['sessionId'];models=[dict(z) for z in db.execute('select model_id,status,tool_call_count from model_usage where session_id=?',(sid,))];meta=[dict(z) for z in db.execute('select tool_name,status,exit_code from tool_usage where session_id=?',(sid,))];count=0
 for row in db.execute('select data from part where session_id=?',(sid,)):
  z=json.loads(row[0]);cmd=z.get('state',{}).get('input',{}).get('command','') if z.get('type')=='tool' else ''
  if f'scripts/execute_frozen.py specs/{tag}.json' in cmd:count+=1
 outer=json.loads((R/f'runs/{tag}-zcode/receipt.json').read_text());inner=json.loads((R/f'runs/{tag}-frozen-controller/receipt.json').read_text());assert count==1 and {x['model_id'] for x in models}=={'deepseek-flash'} and not ({x['tool_name'] for x in meta}&{'Write','Edit'});assert outer['exit_code']==0 and inner['exit_code']=={4:2,7:1}.get(n,0) and inner['inputs_unchanged'];aud.append({'run':tag,'session_id':sid,'model_alias':'deepseek-flash','backend_release':'UNKNOWN','models':models,'tool_metadata':meta,'exact_frozen_command_count':count,'outer_exit':outer['exit_code'],'inner_exit':inner['exit_code'],'inputs_unchanged':True})
db.close();(R/'notes/L031-zcode-audit.json').write_text(json.dumps({'runs':aud,'full_model_IO_published':False,'failed_or_invalid_inner_runs_retained':{'R004':2,'R007':1},'note':'CLI0 is not semantic pass; original failures/invalids and dependency contrast retained'},indent=2)+'\n');print(json.dumps({'audited_runs':len(aud),'all_exact_once':True,'no_direct_Write_Edit':True}))
