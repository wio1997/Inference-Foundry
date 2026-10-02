import json,pathlib,re,hashlib
root=pathlib.Path("/root/ascend/log");patterns=["486286","486679","486738","2824379","2825049","2824506"];rows=[]
for f in root.rglob("*") if root.exists() else []:
 if not f.is_file() or not any(p in f.name for p in patterns):continue
 b=f.read_bytes();ls=b.decode(errors="replace").splitlines();sel=[]
 for n,l in enumerate(ls):
  if ("2026-10-02" in l or "20261002" in l)and any(x in l.lower() for x in["error","timeout","failed","aiv","alltoall","kernel","exception","task"]):
   sel.extend(dict(line=i+1,text=ls[i])for i in range(max(0,n-1),min(len(ls),n+3)))
 rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),selected_lines=list({x["line"]:x for x in sel}.values())[-300:]))
print(json.dumps(dict(root=str(root),files=rows)))
