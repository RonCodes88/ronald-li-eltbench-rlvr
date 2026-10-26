import sys
from pathlib import Path

ELT_BENCH_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ELT-Bench"
# ELT-Bench is not a package: expose its evaluation/ and agents/ modules for import.
sys.path[:0] = [str(ELT_BENCH_DIR / "evaluation"), str(ELT_BENCH_DIR / "agents")]
