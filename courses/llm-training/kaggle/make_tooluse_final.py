"""Build kernel llt-tooluse-final: the CURRENT learner notebook end to end on one T4, full size, with
RUN_EXPERIMENT = True in steps 10 and 11, then step 8 again with USE_OUR_ADAPTER = True (the Hub adapter).
Recording cells are added between steps; nothing else changes. Usage: python make_tooluse_final.py
"""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tooluse_kernels as K

nb = copy.deepcopy(K.nb0)
K.swap_solution(nb)
cells = nb["cells"]
src = K.src
i10 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "SPECIAL = [" in src(c)][0]
i11 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "def make_negatives" in src(c) and not src(c).startswith("%%writefile")][0]
i8 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "USE_OUR_ADAPTER = False" in src(c)][0]
for i in (i10, i11):
    s = src(cells[i])
    assert s.count("RUN_EXPERIMENT = False") == 1
    cells[i]["source"] = s.replace("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")
step8_adapter = copy.deepcopy(cells[i8])
step8_adapter["source"] = src(cells[i8]).replace("USE_OUR_ADAPTER = False", "USE_OUR_ADAPTER = True")

RECORD2 = r'''
# Kaggle validation only: record steps 10 and 11 as printed above.
def pack2(outs, rows_):
    return {"summary": C.summary(rows_), "outputs": outs, "correct": [bool(r["correct"]) for r in rows_],
            "tagReplies": sum("<tool_call>" in o for o in outs)}
res2 = {"step10": {k: pack2(*v) for k, v in ref_results.items()},
        "step11": {**pack2(mix_out, mix), "pairedBootstrapVsBase": C.paired_bootstrap([r["correct"] for r in base], [r["correct"] for r in mix])}}
json.dump(res2, open("/kaggle/working/notebook_final_exp.json", "w"), indent=1, ensure_ascii=False)
print({k: v["summary"]["accuracy"] for k, v in res2["step10"].items()}, res2["step11"]["summary"]["accuracy"])
mix_trainer = None
gc.collect()
torch.cuda.empty_cache()
print("GPU memory allocated before the adapter pass, GB", round(torch.cuda.memory_allocated() / 1e9, 2))
'''
RECORD3 = r'''
# Kaggle validation only: the Hub adapter path of step 8.
res3 = {"summary": C.summary(sft), "outputs": sft_out, "correct": [bool(r["correct"]) for r in sft], "genSec": sft_sec,
        "pairedBootstrapVsBase": C.paired_bootstrap([r["correct"] for r in base], [r["correct"] for r in sft])}
json.dump(res3, open("/kaggle/working/notebook_final_adapter.json", "w"), indent=1, ensure_ascii=False)
print("adapter pass", res3["summary"]["accuracy"], res3["pairedBootstrapVsBase"])
'''
i_rec = i10                                   # RECORD (steps 1 to 9) goes right before step 10
new = cells[:i_rec] + [K.code_cell(K.RECORD)] + cells[i_rec:i11 + 1] + [K.code_cell(RECORD2), step8_adapter, K.code_cell(RECORD3)] + cells[i11 + 1:]
nb["cells"] = new
K.write_kernel("llt-tooluse-final", "LLT ToolUse Final", nb)
print([(src(c)[:40].replace(chr(10), " ")) for c in new if c["cell_type"] == "code"])
