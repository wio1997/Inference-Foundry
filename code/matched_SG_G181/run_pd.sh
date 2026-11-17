set -eu
cd /task/sg_pd_g181
export PYTHONPATH="/task/runtime_sglang_g32:/task/sg_pd_g181${PYTHONPATH:+:$PYTHONPATH}"
exec python3 /task/sg_pd_g181/launch_pd.py
