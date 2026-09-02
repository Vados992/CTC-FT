"""Shared bootstrap for the worked examples."""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def banner(n, title):
    line = "=" * 78
    print(line)
    print(f"EXAMPLE {n}   --    {title}")
    print(line)


def section(t):
    print()
    print(f"--- {t} " + "-" * max(0, 74 - len(t)))
