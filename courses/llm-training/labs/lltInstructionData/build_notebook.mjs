// Builds lltInstructionData.ipynb and steps/ from the lab page's own code blocks, so the notebook, the
// step files and the page cannot disagree. The first two cells come from notebook_head.py (runtime check
// and pinned install), read here as text.
// Usage: node build_notebook.mjs <path-to-lltInstructionData.mjs>
// Check after a build: python check_cells.py (the notebook's code cells must equal cells/*.py, which ran on Kaggle).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = process.argv[2];
if (!src) throw new Error('pass the module file path');
const { lltInstructionData: m } = await import(pathToFileURL(path.resolve(src)).href);

const headPy = fs.readFileSync(path.join(here, 'notebook_head.py'), 'utf-8');
const grab = (name) => {
    const start = headPy.indexOf(name + " = '''") + name.length + 6;
    return headPy.slice(start, headPy.indexOf("'''", start));
};

const cells = [];
const md = (text) => cells.push({ cell_type: 'markdown', metadata: {}, source: text });
const code = (text) => cells.push({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: text });

md(`# ${m.title}\n\n${m.subtitle}\n\nLLM Training Lab, module 14. Regime: ${m.regime.badge}. Needs a T4 GPU (Colab or Kaggle). The cells below are the lab page's code, in order.`);
code(grab('RUNTIME_CHECK'));
code(grab('INSTALL'));

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
    metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' },
                accelerator: 'GPU', colab: { gpuType: 'T4' } },
    nbformat: 4, nbformat_minor: 5,
};
fs.writeFileSync(path.join(here, 'lltInstructionData.ipynb'), JSON.stringify(nb, null, 1) + '\n');
console.log('cells', cells.length, 'code cells from the page', n);
