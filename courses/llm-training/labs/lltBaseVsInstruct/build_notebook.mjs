// Builds lltBaseVsInstruct.ipynb and steps/stepN.py from the lab page's own code blocks,
// so the notebook, the step files and the page can never disagree.
// Usage: node build_notebook.mjs <path-to-lltBaseVsInstruct.mjs>
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = process.argv[2];
if (!src) throw new Error('pass the module file path');
const { lltBaseVsInstruct: m } = await import(pathToFileURL(path.resolve(src)).href);

const RUNTIME_CHECK = `# Runtime check. Run this cell first.
import platform, shutil, sys
ok = True
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    ok = False
    print("Python 3.11 or later is needed, and 3.12 is recommended. In Colab use Runtime > Change runtime type, default runtime.")
try:
    import psutil
    ram = psutil.virtual_memory().total / 1e9
    print("RAM", round(ram, 1), "GB")
    if ram < 6:
        ok = False
        print("This lab holds two 360M models in float32, about 3.3 GB. Close other notebooks or use a machine with 8 GB.")
except ImportError:
    pass
free = shutil.disk_usage(".").free / 1e9
print("Free disk", round(free, 1), "GB")
if free < 3:
    ok = False
    print("The two SmolLM2-360M checkpoints need about 1.5 GB of disk. Free some space.")
try:
    import torch, transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__)
    if int(transformers.__version__.split(".")[0]) < 5:
        print("transformers 4.x returns a list from apply_chat_template. Install 5.x with the next cell.")
except ImportError:
    print("Install the pinned versions with the next cell.")
print("No GPU is needed for this lab." if ok else "Fix the items above, then run this cell again.")`;

const INSTALL = `# Pinned Hugging Face versions of the recorded runs. torch and numpy stay as the runtime ships them. Restart the runtime after installing if transformers was already imported.
!pip install -q "transformers==5.22.1" "tokenizers==0.23.3" "huggingface_hub==2.4.0"`;

const cells = [];
const md = (text) => cells.push({ cell_type: 'markdown', metadata: {}, source: text });
const code = (text) => cells.push({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: text });

md(`# ${m.title}\n\n${m.subtitle}\n\nLLM Training Lab, module 10. Regime: ${m.regime.badge}. Runs on a CPU. The cells below are the lab page's code, in order.`);
code(RUNTIME_CHECK);
code(INSTALL);
const promptsSrc = fs.readFileSync(path.join(here, 'prompts.py'), 'utf-8').trimEnd();
md('The next cell writes `prompts.py`, the 40 prompts that steps 5 and 6 use.');
code('%%writefile prompts.py\n' + promptsSrc);

const stepsDir = path.join(here, 'steps');
fs.mkdirSync(stepsDir, { recursive: true });
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
fs.writeFileSync(path.join(here, 'lltBaseVsInstruct.ipynb'), JSON.stringify(nb, null, 1) + '\n');
console.log('cells', cells.length, 'code cells from the page', n);
