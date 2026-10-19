"""Frozen C32 diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_P_diagnostic_controller_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='66fe89b9672865fa945475181ac9ac9ba95d0fcd6cad224f05c46065169ecd4a'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json','3726d079a2a81898ac809185809cb581c263fa5788b46e10d8022c2371f555dc')
if __name__=='__main__':main()
