"""Frozen C32 L13-D two-step diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_L13_D_diagnostic_controller_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e66dfd6e6b33d3826f37ff1a971f289523f3bf0bd4e0890543a637b5e7edeb6f'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json','035d37267b9895790d0a733a4a219d15d15262382a211064305303b4aa2b9ee2')
if __name__=='__main__':main()
