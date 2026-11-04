"""Build V3 candidate data; never invokes training."""
import argparse,json
from pathlib import Path
from .curriculum import prepare
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--private-seed",type=Path)
    p.add_argument("--v2-export",type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.output,a.private_seed,a.v2_export),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
