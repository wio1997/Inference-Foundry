from pathlib import Path
from collections import defaultdict
import csv,json,re,statistics,sys

def numeric(row,key):
    try:
        return float(row.get(key,0))
    except (TypeError,ValueError):
        return 0.0

def role(row):
    if row.get('Type')!='DispatchFFNCombine':
        return row.get('Type',row.get('Name','unknown'))
    pieces=row.get('Input Shapes','').strip('"').split(';')
    if len(pieces)>3:
        topk=pieces[3].split(',')[-1]
        if topk in ['6','3']:
            return 'DFC_target_top6' if topk=='6' else 'DFC_draft_top3'
    return 'DFC_unknown'

def interval_union_ms(rows):
    intervals=sorted((numeric(row,'Start Time(us)'),numeric(row,'Start Time(us)')+numeric(row,'Duration(us)')) for row in rows)
    total=0.0
    end=0.0
    for start,stop in intervals:
        total+=max(0.0,stop-max(start,end))
        end=max(end,stop)
    return total/1000

root=Path(sys.argv[1]);files=list((root/'traces').rglob('kernel_details.csv'));by_rank=defaultdict(list)
for path in files:
    rank=int(re.search(r'_ep(\d+)_',str(path))[1]);by_rank[rank].append(path)
out={'files':[]}
for rank,paths in sorted(by_rank.items()):
    for index,path in enumerate(sorted(paths,key=lambda p:p.parent.parent.name)):
        stage='early' if index==0 else 'steady'
        groups=defaultdict(list)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                groups[role(row)].append(row)
        results=[]
        for key,rows in groups.items():
            wall=sum(numeric(row,'Duration(us)') for row in rows)
            aic=sum(numeric(row,'aicore_time(us)') for row in rows)
            aiv=sum(numeric(row,'aiv_time(us)') for row in rows)
            results.append({'operator_role':key,'calls':len(rows),'wall_sum_ms':wall/1000,'union_ms':interval_union_ms(rows),'call_median_us':statistics.median(numeric(row,'Duration(us)') for row in rows),'call_p90_us':sorted(numeric(row,'Duration(us)') for row in rows)[int((len(rows)-1)*.9)],'aic_mac_active_pct':sum(numeric(row,'aic_mac_time(us)') for row in rows)/aic*100 if aic else None,'aic_mte2_active_pct':sum(numeric(row,'aic_mte2_time(us)') for row in rows)/aic*100 if aic else None,'aiv_mte2_active_pct':sum(numeric(row,'aiv_mte2_time(us)') for row in rows)/aiv*100 if aiv else None,'first_example':{k:rows[0].get(k) for k in ['Name','Type','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num','Duration(us)','aicore_time(us)','aic_mac_time(us)','aic_mte2_time(us)','aiv_time(us)','aiv_mte2_time(us)']}})
        results.sort(key=lambda r:-r['wall_sum_ms'])
        out['files'].append({'phase':stage,'ep':rank,'dp':0 if rank<8 else 1,'path':str(path),'operators':results})
(root/'kernel-full-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('pmu_complete',len(out['files']))
