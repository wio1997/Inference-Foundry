from pathlib import Path
import json,struct,subprocess,re,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;old=j.parent/"MOONCAKE-ABI-CORE-20261002T0210Z";core=old/"cpu_probe.core";notes=[]
with core.open("rb")as f:
 hdr=f.read(64);assert hdr[:6]==b"\x7fELF\x02\x01";phoff=struct.unpack_from("<Q",hdr,32)[0];phsize,phnum=struct.unpack_from("<HH",hdr,54)
 for i in range(phnum):
  f.seek(phoff+i*phsize);ph=f.read(phsize);typ=struct.unpack_from("<I",ph)[0]
  if typ!=4:continue
  offset=struct.unpack_from("<Q",ph,8)[0];size=struct.unpack_from("<Q",ph,32)[0];f.seek(offset);raw=f.read(size);pos=0
  while pos+12<=len(raw):
   namesz,descsz,kind=struct.unpack_from("<III",raw,pos);pos+=12;name=raw[pos:pos+namesz];pos+=(namesz+3)&~3;desc=raw[pos:pos+descsz];pos+=(descsz+3)&~3
   if kind==0x46494c45:
    count,page=struct.unpack_from("<QQ",desc);paths=desc[16+count*24:].split(bytes([0]));assert len(paths)>=count
    for k in range(count):
     start,end,pages=struct.unpack_from("<QQQ",desc,16+k*24);notes.append({"start":start,"end":end,"file_offset":pages*page,"path":paths[k].decode()})
assert notes
pid=subprocess.check_output(["docker","inspect","-f","{{.State.Pid}}","glm52-single"]).decode().strip();root=Path("/proc")/pid/"root"
stack=(j.parent/"MOONCAKE-ABI-CORE-20261002T0210Z-v2/native_stack.txt").read_text().split("TID 3695240:")[0]
frames=[];symcache={}
for line in stack.splitlines():
 m=re.match(r"#(\d+)\s+(0x[0-9a-f]+)",line)
 if not m:continue
 index=int(m.group(1));addr=int(m.group(2),16)
 if not 4<=index<=19:continue
 maps=[x for x in notes if x["start"]<=addr<x["end"]];assert len(maps)==1
 z=maps[0];elf=root/z["path"].lstrip("/");assert elf.is_file()
 # Verify zero-offset LOAD vaddr0; runtime address maps to ELF vaddr = PC-mapstart+fileoffset.
 with elf.open("rb")as f:
  eh=f.read(64);poff=struct.unpack_from("<Q",eh,32)[0];psz,pnum=struct.unpack_from("<HH",eh,54);zero=False
  for k in range(pnum):
   f.seek(poff+k*psz);p=f.read(psz)
   if struct.unpack_from("<I",p)[0]==1 and struct.unpack_from("<Q",p,8)[0]==0 and struct.unpack_from("<Q",p,16)[0]==0:zero=True
  assert zero,("nonzeroELFzero-offsetLOAD",str(elf))
 off=addr-z["start"]+z["file_offset"]
 if str(elf)not in symcache:
  n=subprocess.run(["nm","-D","-S","-n","-C","--defined-only",str(elf)],capture_output=True,text=True,timeout=30)
  symbols=[]
  for x in n.stdout.splitlines():
   v=re.match(r"([0-9a-f]+)\s+([0-9a-f]+)\s+([tTwW])\s+(.*)",x)
   if v:symbols.append({"address":int(v.group(1),16),"size":int(v.group(2),16),"name":v.group(4)})
  symcache[str(elf)]=symbols
 matches=[x for x in symcache[str(elf)]if x["address"]<=off<x["address"]+x["size"]]
 frames.append({"frame":index,"PC":hex(addr),"module":z["path"],"ELF_offset":hex(off),"contained_dynamic_symbols":matches,"module_sha256":hashlib.sha256(elf.read_bytes()).hexdigest()})
atomic_json(j/"symbolized_frames.json",frames)
def ref(f):
 a=f.read_bytes();return{"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest()}
out={"at":utc(),"native_core":json.loads((old/"reduction.json").read_text())["raw_core"],"symbolized":ref(j/"symbolized_frames.json"),"frames":frames,"signals":0,"engines":0,"inference":0,"limits":["Containsdynamicfunctionsymbol whenPCinsideactualsymbolrange; missingprivatedebugsymbols remainunknown","CoreNT_FILE addressmapping/actualinstalledELFhash, no sourcelevel heapcorruption-origin proof","No newcore/model/API/device actions; existingfrozenCPUcore only"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Actualcore NT_FILE→installedELF address/dynamiccontainedsymbol lookup, no guessednearestprivatefunction;0models/signals/inference","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="existingtaskcore/actualELFsymbolranges",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"symbols":[[x["frame"],x["module"],x["contained_dynamic_symbols"]]for x in frames],"reduction":ref(j/"reduction.json")}))
