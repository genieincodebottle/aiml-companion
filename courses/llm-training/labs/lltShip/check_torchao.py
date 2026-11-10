"""CPU check: the page's fake_quant (step 9) equals torchao's IntxFakeQuantizer, symmetric, group size 32, 2 to 4 bits.
Writes Run Library file lltShip-t1-torchao-check. Needs torchao==0.18.0 (the notebook itself never imports torchao)."""
import sys
from pathlib import Path
import torch, torchao
from torchao.quantization.qat import IntxFakeQuantizeConfig, IntxFakeQuantizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

src = (Path(__file__).resolve().parent / "steps" / "cell09_step9.py").read_text(encoding="utf-8")
ns = {"torch": torch, "BITS": 3, "GROUP": 32}
exec(src.split("def fake_quant")[0].split("BITS, GROUP")[0] + "def fake_quant" + src.split("def fake_quant")[1].split("class FakeQuant")[0], ns)
torch.manual_seed(0)
rows = []
for shape in [(64, 128), (2048, 1024), (3072, 1024)]:
    w = torch.randn(*shape) * 0.02
    for dt, bits in ((torch.int4, 4), (torch.int3, 3), (torch.int2, 2)):
        ref = IntxFakeQuantizer(IntxFakeQuantizeConfig(dt, group_size=32, is_symmetric=True))(w)
        gap = float((ref - ns["fake_quant"](w, bits, 32)).abs().max())
        rows.append({"shape": list(shape), "bits": bits, "maxAbsGap": gap})
        print(shape, bits, gap)
write_run("lltShip-t1-torchao-check", "T1", "baseline", {"torchao": torchao.__version__, "groupSize": 32, "symmetric": True},
          0, rows, notes="fake_quant from steps/cell09_step9.py against torchao IntxFakeQuantizer on random N(0, 0.02) weights")
