"""Frozen C32 diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_P_diagnostic_controller_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='66fe89b9672865fa945475181ac9ac9ba95d0fcd6cad224f05c46065169ecd4a'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v1.json','17d10705007603589013ce991105ac1e60c1aa06385a03bf0f88c459f46543ba')
if __name__=='__main__':main()
