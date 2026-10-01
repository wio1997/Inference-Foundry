import argparse,ast,json,logging,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
class ReplicatedPrefixContract(unittest.TestCase):
 def test_two_prefixes_two_replicas_validate_real_round_logic(self):
  source=Path(__file__).resolve().parents[1]/"adapters/aisbench/prefix_bench.py"
  tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="run_single_round")
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);prefix=root/"prefix.jsonl";full=root/"full.jsonl"
   prefix.write_text("".join(json.dumps({"question":x})+"\n" for x in ["AB","AB","CD","CD"]))
   full.write_text("".join(json.dumps({"question":x})+"\n" for x in ["AB123","CD456","AB789","CDxyz"]))
   accepted=[];phases=[]
   class Collector:
    def __init__(self,*a):pass
    def snapshot(self):return {}
    def compute_hit_rate(self,*a):return {}
    def print_hit_rate_table(self,*a):pass
   def validate(log_dir,phase,count,out):
    self.assertEqual(count,4);accepted.append((phase,count,out))
   namespace={"argparse":argparse,"os":os,"logger":logging.getLogger("prefix-test"),"ROUND_OVERRIDABLE_KEYS":[],
    "parse_prefix_ratio":lambda x:x,"resolve_pod_info":lambda a:[],"HitRateCollector":Collector,
    "create_prefix_dataset":lambda **kw:(str(prefix),str(full)),"prepare_aisbench_dataset_dir":lambda x:t,
    "generate_aisbench_command":lambda *a:[],"modify_aisbench_api":lambda **kw:None,"link_dataset_to_aisbench":lambda *a:None,
    "execute_aisbench_phase":lambda a,p,*args:phases.append(p),"parse_aisbench_log":lambda *a:({},t),
    "validate_phase_details":validate,"archive_log":lambda *a:None,"build_result_row":lambda *a,**kw:{},
    "write_csv":lambda *a:None,"write_jsonl":lambda *a:None}
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
   args=argparse.Namespace(dataset=None,dataset_path=t,model_path="unused",input_len=5,repeat_rate=.4,data_num=4,dp=2,prefix_num=2,seed=1,length_mean=None,length_std=None,length_min=None,length_max=None,work_path=t,summarizer="default_perf",output_dir=t,model_name="fake",host_ip="none",host_port=0,url="",request_rate=0,test_type="stream",enable_think=False,npu_num=1,result_csv="unused",result_jsonl="unused",concurrency=2,output_len=3)
   fake=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:types.SimpleNamespace(encode=lambda text,**kw:list(text))))
   previous=os.getcwd()
   try:
    os.chdir(t)
    with patch.dict(sys.modules,{"transformers":fake}):namespace["run_single_round"](args,1)
   finally:os.chdir(previous)
   self.assertEqual(phases,["warmup","full"]);self.assertEqual(accepted,[("warmup",4,1),("full",4,3)])
   self.assertTrue(json.loads((root/"dataset_validation.json").read_text())["valid"])
if __name__=="__main__":unittest.main()
