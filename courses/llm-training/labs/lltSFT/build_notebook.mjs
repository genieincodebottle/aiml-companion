// Builds lltSFT.ipynb and steps/cellNN_stepK.py from the lab page's own code blocks, so the notebook, the step
// files and the page cannot disagree. The notebook adds a runtime check, the pinned install and a cell that writes
// sft_common.py (the same selection and mask, plus the evaluation helpers step 8 imports).
// Usage: node build_notebook.mjs <path-to-lltSFT.mjs>
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = process.argv[2];
if (!src) throw new Error('pass the module file path');
const mod = await import(pathToFileURL(path.resolve(src)).href);
const m = mod.lltSFT || mod.default;

const RUNTIME_CHECK = `# Runtime check. Run this cell first.
import os, platform, shutil, subprocess, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # before torch loads: SFTTrainer refuses a model split over 2 GPUs
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("PROBLEM Python 3.11 or later is needed, 3.12 recommended. Start a current Colab or Kaggle session.")
free = shutil.disk_usage(".").free / 1e9
print("Free disk", round(free, 1), "GB")
if free < 5:
    print("PROBLEM steps 1 to 8 download about 1.3 GB of models and stream the data. Free some disk.")
try:   # a CPU-only runtime has no nvidia-smi at all
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = ""
print("GPU", gpu or "none")
if not gpu:
    print("Steps 1 to 7 run on this CPU. Step 8 needs a GPU: Colab, Runtime > Change runtime type > T4 GPU. Kaggle, Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    print("PROBLEM a P100 has no kernels in current PyTorch. Pick a T4 instead.")
from importlib.metadata import version, PackageNotFoundError
for pkg, want in [("transformers", "5.22.1"), ("trl", "1.17.1"), ("accelerate", "1.15.0"), ("datasets", "5.1.0")]:
    try:
        have = version(pkg)
    except PackageNotFoundError:
        have = "not installed"
    print(pkg, have, "" if have == want else "(the next cell installs " + want + ")")
print("Runtime OK" if sys.version_info >= (3, 11) else "Fix the PROBLEM lines, then run this cell again.")`;

const INSTALL = `# Pinned Hugging Face versions of our recorded runs. torch and numpy stay as the runtime ships them.
# If transformers was already imported in this session, restart the session after this cell, then run from step 1.
%pip install -q "transformers==5.22.1" "trl==1.17.1" "accelerate==1.15.0" "datasets==5.1.0" "tokenizers==0.23.3" "huggingface_hub==2.4.0"`;

const cells = [];
const md = (text) => cells.push({ cell_type: 'markdown', metadata: {}, source: text });
const code = (text) => cells.push({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: text });

md(`# ${m.title}\n\n${m.subtitle}\n\nLLM Training Lab, module 11. Regime: ${m.regime.badge}. Steps 1 to 7 run on a CPU with SmolLM2-135M. Step 8 fine-tunes SmolLM2-360M on a T4. The cells below are the lab page's code, in order.`);
code(RUNTIME_CHECK);
code(INSTALL);
const helper = fs.readFileSync(path.join(here, 'sft_common.py'), 'utf-8').trimEnd();
// The page's cross_doc_collator (step 10) must be the exact function the Run Library runs imported from sft_common.py.
const allCode = m.lesson.build.flatMap((s) => s.blocks).filter((b) => typeof b.code === 'string').map((b) => b.code).join('\n');
const fnStart = allCode.indexOf('def cross_doc_collator');
if (fnStart < 0) throw new Error('page has no cross_doc_collator');
const fnText = allCode.slice(fnStart, allCode.indexOf('    return wrapped', fnStart) + '    return wrapped'.length);
if (!helper.includes(fnText)) throw new Error('page cross_doc_collator differs from sft_common.py');
for (const ch of allCode) if (ch.charCodeAt(0) > 127) throw new Error('non-ASCII character in page code: ' + ch);
md('The next cell writes `sft_common.py`. It holds the same selection and label mask as steps 2 and 3, and the two evaluation helpers that step 8 imports.');
code('%%writefile sft_common.py\n' + helper);

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

const toLines = (s) => s.split('\n').map((l, k, a) => (k < a.length - 1 ? l + '\n' : l));
const nb = {
    cells: cells.map((c) => ({ ...c, source: toLines(c.source) })),
    metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' } },
    nbformat: 4, nbformat_minor: 5,
};
fs.writeFileSync(path.join(here, 'lltSFT.ipynb'), JSON.stringify(nb, null, 1) + '\n');
console.log('cells', cells.length, 'code cells from the page', n);
