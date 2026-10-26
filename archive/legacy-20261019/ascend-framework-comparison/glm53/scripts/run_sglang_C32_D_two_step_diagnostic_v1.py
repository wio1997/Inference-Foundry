"""Frozen C32 diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_D_diagnostic_controller_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='397997aae778814836429fa1e6f2cd57ef242768ea06223e41ac299b4c0650a8'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-only-L04-restore/D-two-step-diagnostic-v1.json','f2ab656c9e4b5fde35448decbff6b13cff209f5b76fa91a0577bbb91b73efc60')
if __name__=='__main__':main()
