"""Pure control guards; matching IDs or a prepared slot are not authorization."""
import hashlib,json
def approved(plan_bytes,prepared_bytes,approval):
    plan=json.loads(plan_bytes)
    if approval.get('status')!='AUTHORIZED_ONCE' or not approval.get('user_reply'):
        raise RuntimeError('Fresh full-model window approval missing')
    if approval.get('plan_sha256')!=hashlib.sha256(plan_bytes).hexdigest():
        raise RuntimeError('Approved plan bytes differ')
    if approval.get('prepared_sha256')!=hashlib.sha256(prepared_bytes).hexdigest():
        raise RuntimeError('Prepared container/kit receipt differs')
    if approval.get('approval_id')!=plan['approval_slot_id']:
        raise RuntimeError('Approval slot mismatch')
    return plan
def owns_processes(busy,owned):
    if not busy or not set(busy)<=set(owned):raise RuntimeError('NPU ownership incomplete or foreign process present')
def restore_allowed(busy):return not busy
def no_existing_submission(path):
    if path.exists():raise RuntimeError('Existing submission: inspect its handle;never resubmit from a polling timeout')
def idle_metrics(text):
    rows=[x for x in text.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
    if len(rows)!=4 or any(float(x.rsplit(None,1)[1])!=0 for x in rows):
        raise RuntimeError('Both DP engines must have zero running/waiting requests')
    return rows
