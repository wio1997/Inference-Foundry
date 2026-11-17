set -eu
cd /task/sg_pd_g164
export PYTHONPATH="/task/runtime_sglang_g32:/task/sg_pd_g164${PYTHONPATH:+:$PYTHONPATH}"
exec python3 /task/sg_pd_g164/launch_pd.py
