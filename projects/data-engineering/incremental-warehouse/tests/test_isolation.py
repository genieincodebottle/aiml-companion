"""The answer key never reaches the build path."""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "warehouse"
ALLOWED = {"world.py", "scoring.py"}
SYMBOL = re.compile(r"answer_key|AnswerKey", re.IGNORECASE)


def test_only_the_generator_and_the_scorer_name_the_answer_key():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
                 if p.name not in ALLOWED and SYMBOL.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_pipeline_modules_do_not_import_the_generator_or_the_scorer():
    pipeline = ["checks", "clock", "dag", "extract", "land", "load", "options", "pipeline",
                "transform", "warehouse"]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib\n"
        "for m in %r: importlib.import_module('warehouse.' + m)\n"
        "bad = [m for m in ('warehouse.scoring', 'warehouse.source.world') if m in sys.modules]\n"
        "print(','.join(bad))\n" % (str(SRC.parent), pipeline))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_pipeline_modules_never_read_the_wall_clock():
    banned = re.compile(r"date\.today|datetime\.now|datetime\.utcnow|time\.time\(")
    for name in ("extract", "checks", "load", "transform", "pipeline", "land"):
        text = (SRC / f"{name}.py").read_text(encoding="utf-8")
        assert not banned.search(text), f"{name}.py reads the wall clock"
