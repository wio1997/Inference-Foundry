#!/usr/bin/env python3
"""Create a concrete reviewed controller from the proven CPU runner before freezing it."""
import argparse,ast,hashlib,json,pathlib,re
ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--case-key',required=True);args=ap.parse_args();assert re.fullmatch(r'L011-R\d{3}',args.run_id)
R=pathlib.Path(__file__).resolve().parents[1];proto=json.loads((R/'specs/L011-protocol.json').read_text());assert args.case_key in proto['cases'];manifest=json.loads((R/proto['input_manifest']).read_text());by={f['label']:f for f in manifest['fixtures']};extra=['specs/L011-protocol.json',proto['input_manifest'],*[by[n]['path'] for n in proto['cases'][args.case_key]]]
base=R/'scripts/run_tokenizer_image_cpu_v3.py';s=base.read_text().replace('L010-R006',args.run_id).replace('l010-cpu','l011-cpu').replace('20261019-r006','20261019-'+args.run_id.split('-')[1].lower()).replace('probe_tokenizer_image.py','bench_tokenizer_image.py')
marker="bootstrap=strings['BOOTSTRAP']";assert s.count(marker)==1;where=s.index(marker);injection="extra_paths="+repr(extra)+"\npayload['extra_inputs']={f:{'content':(R/f).read_text(),'sha256':hashlib.sha256((R/f).read_bytes()).hexdigest()} for f in extra_paths}\npayload['case_key']="+repr(args.case_key)+"\n";s=s[:where]+injection+s[where:]
block=r'''EXTRA_BOOTSTRAP=r"""
for filename,item in j['extra_inputs'].items():
 assert not filename.startswith('/') and '..' not in pathlib.PurePosixPath(filename).parts
 p=source/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['content']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
sys.argv+=['--case-key',j['case_key'],'--protocol',str(source/'specs/L011-protocol.json'),'--fixture-root',str(source)]
"""
needle="runpy.run_path(str(script),run_name='__main__')";assert bootstrap.count(needle)==1;bootstrap=bootstrap.replace(needle,EXTRA_BOOTSTRAP+'\n'+needle)
'''
where=s.index('VENV_BOOTSTRAP=r');s=s[:where]+block+s[where:];ast.parse(s)
path=R/'scripts'/('run_'+args.run_id.replace('-','_')+'_tokenizer.py');path.open('x').write(s)
metadata={'run_id':args.run_id,'case_key':args.case_key,'base_controller_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'materializer_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'generated_controller':str(path.relative_to(R)),'generated_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'extra_inputs':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in extra},'note':'Same CPU image isolation/model mounts/package view as L010-R006; new clean benchmark probe and frozen per-case inputs only.'};out=R/'specs'/(args.run_id+'-build.json');out.open('x').write(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))
