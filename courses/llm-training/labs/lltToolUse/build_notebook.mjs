// Builds lltToolUse.ipynb and steps/cellNN_stepK.py from the lab page's own code blocks, so the notebook, the step
// files and the page cannot disagree. The notebook adds a runtime check, the pinned install and a cell that writes
// tool_common.py (the shared helpers the steps import as C).
// Usage: node build_notebook.mjs <path-to-lltToolUse.mjs>
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = process.argv[2];
if (!src) throw new Error('pass the module file path');
const mod = await import(pathToFileURL(path.resolve(src)).href);
const m = mod.lltToolUse || mod.default;

const RUNTIME_CHECK = `# Runtime check. Run this cell first. It imports nothing heavy, so it works before the install cell.
import os, platform, shutil, subprocess, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")                       # before torch loads: one GPU for Trainer
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("PROBLEM Python 3.11 or later is needed, 3.12 recommended. Start a current Colab or Kaggle session.")
free = shutil.disk_usage(".").free / 1e9
print("Free disk", round(free, 1), "GB")
if free < 12:
    print("PROBLEM the models need about 9 GB of downloads (Qwen3.5-2B-Base, Qwen3.5-0.8B-Base, Qwen3-1.7B-Base). Free some disk.")
try:   # a CPU-only runtime has no nvidia-smi at all
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = ""
print("GPU", gpu or "none")
if not gpu:
    print("Steps 1 to 6 run on this CPU. Steps 7 to 11 need a GPU: Colab, Runtime > Change runtime type > T4 GPU. Kaggle, Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    print("PROBLEM a P100 has no kernels in current PyTorch. Pick a T4 instead.")
from importlib.metadata import version, PackageNotFoundError
for pkg, want in [("transformers", "5.22.1"), ("trl", "1.17.1"), ("peft", "0.21.2"), ("accelerate", "1.15.0"), ("datasets", "5.1.0"), ("bfcl-eval", "2026.3.23")]:
    try:
        have = version(pkg)
    except PackageNotFoundError:
        have = "not installed"
    print(pkg, have, "" if have == want else "(the next cell installs " + want + ")")
print("Runtime OK" if sys.version_info >= (3, 11) else "Fix the PROBLEM lines, then run this cell again.")`;

const INSTALL = `# Pinned versions of our recorded runs. torch and numpy stay as the runtime ships them.
# torchao is removed because PEFT 0.21.2 refuses the version on the Kaggle image.
# bfcl-eval is installed without its dependencies: they pin numpy 1.26.4, which would break the runtime's torch.
# If transformers was already imported in this session, restart the session after this cell, then run from step 1.
%pip uninstall -y -q torchao
%pip install -q "transformers==5.22.1" "trl==1.17.1" "peft==0.21.2" "accelerate==1.15.0" "datasets==5.1.0" "tokenizers==0.23.3" "huggingface_hub==2.4.0"
%pip install -q --no-deps "bfcl-eval==2026.3.23"`;

const cells = [];
const md = (text) => cells.push({ cell_type: 'markdown', metadata: {}, source: text });
const code = (text) => cells.push({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: text });

md(`# ${m.title}\n\n${m.subtitle}\n\nLLM Training Lab, module 16. Regime: ${m.regime.badge}. Steps 1 to 6 run on a CPU. Steps 7 to 11 need one T4. The cells below are the lab page's code, in order.`);
code(RUNTIME_CHECK);
code(INSTALL);
const helper = fs.readFileSync(path.join(here, 'tool_common.py'), 'utf-8').split(String.fromCharCode(13)).join('').trimEnd();
const allCode = m.lesson.build.flatMap((s) => s.blocks).filter((b) => typeof b.code === 'string').map((b) => b.code).join('\n');
// Functions the page defines must be the exact text of tool_common.py, which the Kaggle runs import.
for (const fn of ['def hermes_to_chat', 'def build_labels', 'def to_json_schema', 'def coerce', 'def parse_xml_calls', 'def train_lora']) {
    const start = allCode.indexOf(fn);
    if (start < 0) throw new Error('page has no ' + fn);
    const rest = allCode.slice(start);
    const endRel = rest.search(/\n(?=\S)/);
    const fnText = endRel < 0 ? rest : rest.slice(0, endRel);
    if (!helper.includes(fnText.trimEnd())) throw new Error('page ' + fn + ' differs from tool_common.py');
}
for (const ch of allCode + helper) if (ch.charCodeAt(0) > 127) throw new Error('non-ASCII character in code: ' + ch);
md('The next cell writes `tool_common.py`, the helpers the steps import as `C`: the data selection, the BFCL loader, the parsers, the checker, the constrained decoder and the generation loop. Steps 2 to 8 show the functions you build yourself.');
code('%%writefile tool_common.py\n' + helper);

const stepsDir = path.join(here, 'steps');
fs.mkdirSync(stepsDir, { recursive: true });
for (const f of fs.readdirSync(stepsDir)) fs.unlinkSync(path.join(stepsDir, f));
let n = 0;
m.lesson.build.forEach((step, i) => {
    const blocks = step.blocks.filter((b) => typeof b.code === 'string');
    md(`## Step ${i + 1}. ${step.heading}\n\n${step.expect ? `What you should see. ${step.expect}` : ''}`);
    blocks.forEach((b) => {
        n += 1;
        code(b.code);
        fs.writeFileSync(path.join(stepsDir, `cell${String(n).padStart(2, '0')}_step${i + 1}.py`), b.code + '\n');
    });
});
m.challenges.forEach((c, i) => {
    md(`## Prove it ${i + 1}\n\n${c.prompt}`);
    code(c.starter);
    fs.writeFileSync(path.join(stepsDir, `challenge${i + 1}_starter.py`), c.starter + '\n');
    fs.writeFileSync(path.join(stepsDir, `challenge${i + 1}_solution.py`), c.solution + '\n');
});
const nb = {
    cells,
    metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' }, accelerator: 'GPU', colab: { gpuType: 'T4' } },
    nbformat: 4, nbformat_minor: 5,
};
fs.writeFileSync(path.join(here, 'lltToolUse.ipynb'), JSON.stringify(nb, null, 1));
console.log('wrote lltToolUse.ipynb with', cells.length, 'cells and', n, 'step code cells');
