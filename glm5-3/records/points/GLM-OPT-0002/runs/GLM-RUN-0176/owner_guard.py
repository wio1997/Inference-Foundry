import json,fcntl,time,os,threading
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).parent;task=Path("/data/tiankuan/wio/glm52-pd")
owner=json.loads((task/"controller-owner.json").read_text())
def guard():
 c=json.loads((task/"controller-owner.json").read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(c["heartbeat_at"])).total_seconds()
 assert c["run_id"]==root.name and c["status"]=="running" and c["owner"]==owner["owner"] and -2<=age<=30
 assert c["owner"]["boot_id"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip()
 for n in [".controller.lock",".formal-test.lock"]:
  with (task/n).open("a+") as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError("unique controller lock lost")
def start_watchdog():
 guard()
 def watch():
  while True:
   time.sleep(2)
   try:guard()
   except BaseException:os._exit(70)
 threading.Thread(target=watch,daemon=True).start()
