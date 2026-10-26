#!/usr/bin/env python3
import json,pathlib,sqlite3
R=pathlib.Path(__file__).resolve().parents[1];db=sqlite3.connect('file:/Users/wio/.zcode/cli/db/db.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row;rows=[]
for n in range(1,10):
 tag=f'L034-R{n:03}';sid=json.loads((R/f'runs/{tag}-zcode/stdout.jsonl').read_text())['sessionId'];models=[dict(x) for x in db.execute('select model_id,status,tool_call_count from model_usage where session_id=?',(sid,))];meta=[dict(x) for x in db.execute('select tool_name,status,exit_code from tool_usage where session_id=?',(sid,))];count=0
 for row in db.execute('select data from part where session_id=?',(sid,)):
  x=json.loads(row[0]);cmd=x.get('state',{}).get('input',{}).get('command','') if x.get('type')=='tool' else ''
  if f'scripts/execute_frozen.py specs/{tag}.json' in cmd:count+=1
 outer=json.loads((R/f'runs/{tag}-zcode/receipt.json').read_text());inner=json.loads((R/f'runs/{tag}-frozen-controller/receipt.json').read_text());assert count==1 and {x['model_id'] for x in models}=={'deepseek-flash'} and not ({x['tool_name'] for x in meta}&{'Write','Edit'});expected=0 if n in [1,2,3,9] else 1;assert outer['exit_code']==0 and inner['exit_code']==expected and inner['inputs_unchanged'];rows.append({'run':tag,'session_id':sid,'model_alias':'deepseek-flash','backend_release':'UNKNOWN','models':models,'tool_metadata':meta,'exact_frozen_command_count':count,'outer_exit':outer['exit_code'],'inner_exit':inner['exit_code'],'inputs_unchanged':True,'semantic_scope':'INVALID harness' if 4<=n<=7 else 'conditional identity contract FAIL' if n==8 else 'scoped PASS'})
db.close();(R/'notes/L034-zcode-audit.json').write_text(json.dumps({'runs':rows,'full_model_IO_published':False,'CLI0_is_not_semantic_success':True},indent=2)+'\n');print(json.dumps({'runs':len(rows),'exact_once':True,'invalids_retained':4,'conditional_failures_retained':1}))
