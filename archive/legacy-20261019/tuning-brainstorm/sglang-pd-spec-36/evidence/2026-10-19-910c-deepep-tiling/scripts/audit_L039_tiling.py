"""Metadata-only audit of exact task execution sessions; never publish model I/O."""
import datetime,json,pathlib,shlex,sqlite3
R=pathlib.Path(__file__).resolve().parents[1];db=sqlite3.connect('file:/Users/wio/.zcode/cli/db/db.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row;rows=[]
for n in range(46,48):
 tag=f'L039-R{n:03}';outer=json.loads((R/f'runs/{tag}-zcode/receipt.json').read_text());inner=json.loads((R/f'runs/{tag}-frozen-controller/receipt.json').read_text());sid=json.loads((R/f'runs/{tag}-zcode/stdout.jsonl').read_text())['sessionId']
 models=[dict(x) for x in db.execute('select model_id,status,tool_call_count from model_usage where session_id=?',(sid,))];meta=[dict(x) for x in db.execute('select tool_name,status,exit_code from tool_usage where session_id=?',(sid,))];count=0
 for row in db.execute('select data from part where session_id=?',(sid,)):
  x=json.loads(row[0]);cmd=x.get('state',{}).get('input',{}).get('command','') if x.get('type')=='tool' else '';lexer=shlex.shlex(cmd,posix=True,punctuation_chars=True);lexer.whitespace_split=True;tokens=list(lexer)
  count+=sum(tokens[i] in ('python3','python') and tokens[i+1:i+3]==['scripts/execute_frozen.py',f'specs/{tag}.json'] for i in range(max(0,len(tokens)-2)))
 assert count==1 and {x['model_id'] for x in models}=={'deepseek-flash'} and not ({x['tool_name'] for x in meta}&{'Write','Edit'}),tag
 assert outer['exit_code']==0 and inner['inputs_unchanged'],tag
 rows.append({'run':tag,'session_id':sid,'model_alias':'deepseek-flash','backend_release':'UNKNOWN','tool_metadata':meta,'exact_frozen_command_count':count,'outer_exit':outer['exit_code'],'inner_exit':inner['exit_code'],'inputs_unchanged':True})
db.close();o={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':rows,'full_model_IO_published':False,'user_credentials_in_executor_prompt':False,'CLI0_is_not_semantic_success':True,'ephemeral_transfer_TLS_keys':'Generated/used only by machines on task-owned remote jobs; no private material in prompts/results/publication.'};(R/'notes/L039-tiling-zcode-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'sessions':len(rows),'commands_exact_once':len(rows),'inner_failures':[x['run'] for x in rows if x['inner_exit']!=0],'full_IO_published':False}))
