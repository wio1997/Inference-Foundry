import hashlib,json,random,re
from pathlib import Path
from transformers import PreTrainedTokenizerFast
R=Path('/task');tokenizer=PreTrainedTokenizerFast.from_pretrained(R/'tokenizer');special=set(tokenizer.all_special_ids);words=[]
for value in tokenizer.get_vocab().values():
 if value in special:continue
 piece=tokenizer.decode([value],clean_up_tokenization_spaces=False)
 if re.fullmatch(r' [A-Za-z]{2,18}',piece) and len(tokenizer.encode(piece,add_special_tokens=False))==1:words.append(piece)
words=sorted(set(words));assert len(words)>100
probe=''.join(random.Random(20261112).choices(words,k=2048));assert len(tokenizer.encode(probe,add_special_tokens=True))==2048
records=[]
template='''from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
models=[dict(attr='service',type=VLLMCustomAPI,abbr='sglang-a3-dsv41-1017-w8a8-engram-dram',path='/task/tokenizer',model='Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8',stream=True,request_rate=0,retry=1,host_ip='127.0.0.1',host_port=18167,max_out_len=1024,batch_size=16,trust_remote_code=True,generation_kwargs=dict(temperature=0.0,ignore_eos=True,min_tokens=1024))]
datasets=[dict(abbr='PHASE_input_SIZE_output1024',type=CustomDataset,path='/task/datasets/PHASE_input_SIZE.jsonl',meta_path='',reader_cfg=dict(input_columns=['question','max_out_len'],output_column='answer'),infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template='{question}'),retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer)),eval_cfg=dict(evaluator=dict(type=AccEvaluator),pred_role='BOT'))]
summarizer=dict(attr='performance',type=DefaultPerfSummarizer,calculator=dict(type=DefaultPerfMetricCalculator,stats_list=['Average','Min','Max','Median','P75','P90','P99']))
'''
for size in [32768,65536,131072,262144]:
 for phase,count,seed in [('warmup',16,20261112+size+999),('formal',32,20261112+size)]:
  rng=random.Random(seed);path=R/f'datasets/{phase}_input_{size}.jsonl';lengths=[]
  with path.open('w') as out:
   for i in range(count):
    text=''.join(rng.choices(words,k=size));actual=len(tokenizer.encode(text,add_special_tokens=True));assert actual==size,(phase,size,i,actual)
    row={'question':text,'answer':'','max_out_len':1024,'input_tokens':actual,'request_id':f'{phase}-{size}-{i}'};out.write(json.dumps(row,ensure_ascii=False)+'\n');lengths.append(actual)
  config=R/f'configs/aisbench_{phase}_{size}.py';config.write_text(template.replace('PHASE',phase).replace('SIZE',str(size)))
  records.append({'phase':phase,'input_tokens':size,'requests':count,'min_actual':min(lengths),'max_actual':max(lengths),'seed':seed,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'config_path':str(config)});print(json.dumps(records[-1]),flush=True)
manifest={'model':'Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8','output_tokens':1024,'concurrency':16,'lexicon':len(words),'tokenizer_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'tokenizer').glob('*.json') if p.name!='source_manifest.json'},'prefix_policy':'Distinct warmup/formal texts; planned SGLang cache flush after warmup and before counted run; radix remains enabled; actual hits require runtime evidence','profiles':records};(R/'datasets/manifest.json').write_text(json.dumps(manifest,indent=2))
from mmengine.config import Config
from ais_bench.benchmark.cli.config_manager import CustomConfigChecker
from ais_bench.benchmark.utils.config.run import try_fill_in_custom_cfgs
for p in records:
 c=try_fill_in_custom_cfgs(Config.fromfile(p['config_path'],format_python_code=False));CustomConfigChecker(c,p['config_path']).check();assert c.models[0].batch_size==16;assert c.datasets[0].infer_cfg.prompt_template.template=='{question}'
(R/'benchmark_preparation_validation.json').write_text(json.dumps({'phase':'datasets_and_official_configs_validated','exact_prompts':192,'profiles':8,'api_verified':False,'performance_measurements':False},indent=2));print('Exact final lengths validated for all192 prompts; all8 official configs checked. No API inference or performance results.',flush=True)
