#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--limit',type=int,default=8);args=p.parse_args()
root=Path(__file__).resolve().parents[1];q=args.query.casefold()
facts=[json.loads(x) for x in (root/'knowledge/facts.jsonl').open()]
questions=json.loads((root/'knowledge/questions.json').read_text())['questions']
hits=[x for x in facts+questions if q in json.dumps(x,ensure_ascii=False).casefold()]
print(json.dumps({'query':args.query,'total':len(hits),'results':hits[:args.limit],
 'note':'先核对valid_when/失效条件；历史运行值不是新观测。无新输入或新增待裁决结论，不重开相同取证。'},ensure_ascii=False,indent=2))
