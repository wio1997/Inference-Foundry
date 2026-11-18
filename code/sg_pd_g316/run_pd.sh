set -eu
cd /task/sg_pd_g316
export PYTHONPATH="/task/runtime_sglang_g32:/task/sg_pd_g316${PYTHONPATH:+:$PYTHONPATH}"
exec python3 /task/sg_pd_g316/launch_pd.py
