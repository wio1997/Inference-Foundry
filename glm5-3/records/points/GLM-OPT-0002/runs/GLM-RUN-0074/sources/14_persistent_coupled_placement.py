
"""Opt-in durable coupled fault quarantine.

One gateway writer holds a lifetime flock. A durable open marker makes an
unclean gateway exit conservative for unchanged native owner epochs. Healthy
lease-free shutdown clears that marker, never a recorded native fault. New
physical owner epochs are supplied by the unique task controller, not HTTP.
Native payloads, existing leases and execution are unchanged.
"""
import fcntl,json,math,os,uuid
from pathlib import Path
from coupled_placement import CoupledPlacement

class PersistentCoupledPlacement(CoupledPlacement):
    def __init__(self,replicas,policy="active_count",groups=None,fault_state_path=None):
        super().__init__(replicas,policy,groups)
        self.journal_error=None;self.journal_fd=None;self.journal_path=None
        if fault_state_path is None:return
        if not self.execution_groups:raise ValueError("fault journal requires declared native groups")
        p=Path(fault_state_path)
        if not p.is_absolute()or not p.parent.is_dir()or p.is_symlink():raise ValueError("existing absolute task journal directory required")
        self.journal_path=p
        fd=os.open(str(p)+".lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            prior=json.loads(p.read_text())if p.exists()else None
            if prior is not None:
                if not isinstance(prior,dict)or prior.get("schema")!=1 or type(prior.get("open"))is not bool or not isinstance(prior.get("groups"),dict):
                    raise ValueError("invalid durable group state; no admissions")
                for key,row in prior["groups"].items():
                    if not isinstance(row,dict)or not isinstance(row.get("epoch"),str)or not row["epoch"]or type(row.get("faulted"))is not bool or not isinstance(row.get("members"),list)or any(not isinstance(x,str)for x in row["members"]):
                        raise ValueError("invalid durable group entry; no admissions")
                    group=self.execution_groups.get(key)
                    if group is not None and group.epoch==row["epoch"]:
                        if sorted(group.members)!=sorted(row["members"]):raise ValueError("same native epoch group membership drift")
                        group.faulted=row["faulted"]or prior["open"]
            self.journal_fd=fd
            for group in self.execution_groups.values():
                if group.faulted:
                    for key in group.members:self.replicas[key].unhealthy_until=math.inf
            # Persist before an app can admit requests. Dirty state is conservative
            # even if a process exits before persisting a newly observed failure.
            self._persist(True)
        except BaseException:
            self.journal_fd=None;os.close(fd);raise

    def _persist(self,opened):
        p=self.journal_path
        document={"schema":1,"open":opened,"groups":{g.key:{"epoch":g.epoch,"members":sorted(g.members),"faulted":g.faulted}for g in self.execution_groups.values()}}
        tmp=p.parent/("."+p.name+"."+uuid.uuid4().hex)
        try:
            fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,"w")as stream:
                json.dump(document,stream,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,p)
            directory=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if tmp.exists():tmp.unlink()

    async def release(self,lease,backend_failure=False):
        released=await super().release(lease,backend_failure)
        if released and backend_failure and self.journal_fd is not None and lease.replica.key in self.member_group:
            async with self.lock:
                try:self._persist(True)
                except OSError as error:
                    # Lease and in-memory whole-group quarantine are already
                    # committed; durable open marker remains conservative.
                    self.journal_error=repr(error)
        return released

    async def close(self):
        if self.journal_fd is None:return
        async with self.lock:
            fd=self.journal_fd
            try:
                if self.journal_error is None:self._persist(bool(self.leases))
            finally:self.journal_fd=None;os.close(fd)

    async def snapshot(self):
        rows=await super().snapshot()
        for row in rows:
            row["fault_journal_enabled"]=self.journal_path is not None
            row["fault_journal_error"]=self.journal_error
        return rows
