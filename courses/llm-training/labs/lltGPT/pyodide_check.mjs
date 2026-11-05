// Run the lab page's browser cells (steps/step1.py, the challenge starter and solution) in real
// Pyodide 0.27.0, one fresh __main__ namespace each, and write runs/lltGPT/lltGPT-t0-pyodide.json.
// Exact output only; Pyodide timings go to measure_timing.py's idle-machine pass if ever needed.
// Setup once in any folder: npm i pyodide@0.27.0. Run: NODE_PATH=<folder>/node_modules node pyodide_check.mjs <lab dir>
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(path.join(process.env.NODE_PATH || '.', 'x.js'));
const { loadPyodide } = require('pyodide');
const lab = process.argv[2];
const py = await loadPyodide();
await py.loadPackage('numpy');
const cells = {};
for (const name of ['step1', 'challenge_starter', 'challenge_solution']) {
    let out = '';
    py.setStdout({ batched: (m) => { out += m + '\n'; } });
    py.setStderr({ batched: (m) => { out += '[stderr] ' + m + '\n'; } });
    const g = py.runPython("__import__('types').ModuleType('__main__').__dict__");
    try {
        await py.runPythonAsync(fs.readFileSync(path.join(lab, 'steps', `${name}.py`), 'utf8'), { globals: g });
    } catch (e) {
        out += '[exception] ' + String(e.message).trim().split('\n').slice(-1)[0] + '\n';
    }
    g.destroy();
    cells[name] = out;
    console.log('=====', name); console.log(out);
}
const rec = {
    runId: 'lltGPT-t0-pyodide', module: 'lltGPT', tier: 'T0', label: 'baseline',
    hardware: { gpu: null, count: 0, vramGB: null }, env: { pyodide: py.version, numpy: py.runPython('import numpy; numpy.__version__') },
    notebookCommit: null, config: { cells: Object.keys(cells) }, seed: 0, metrics: [], evals: { stdout: cells },
    samples: [], wallClockSec: null, costUSD: 0, priceCheckedOn: null, date: new Date().toISOString().slice(0, 10),
};
fs.writeFileSync(path.join(lab, '..', '..', 'runs', 'lltGPT', 'lltGPT-t0-pyodide.json'), JSON.stringify(rec, null, 1));
console.log('wrote lltGPT-t0-pyodide.json');
