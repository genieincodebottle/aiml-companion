// Run the lab page's browser cells (cells/step1.py to step3.py and the challenge solution) in real
// Pyodide 0.27.0, the version the app pins, the way the page runs them: numpy loaded, one fresh
// __main__ namespace per run, 3 runs each. Writes ../../runs/lltAttention/lltAttention-t0-pyodide.json.
//
// Setup (once, in any empty folder):  npm i pyodide@0.27.0
// Run: copy this file into that folder and run it there with the lab path as the argument:
//       node pyodide_time.mjs <path to labs/lltAttention>
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadPyodide } from 'pyodide';

const lab = process.argv[2];
const runsDir = path.join(lab, '..', '..', 'runs', 'lltAttention');
const t0 = Date.now();
const py = await loadPyodide();
await py.loadPackage(['numpy']);
const loadMs = Date.now() - t0;
const steps = {};
for (const name of ['step1', 'step2', 'step3', 'challenge_solution']) {
    const code = fs.readFileSync(path.join(lab, 'cells', `${name}.py`), 'utf8');
    const times = [];
    let first = null;
    for (let r = 0; r < 3; r++) {
        let out = '';
        py.setStdout({ batched: (m) => { out += m + '\n'; } });
        py.setStderr({ batched: (m) => { out += '[Error] ' + m + '\n'; } });
        const g = py.runPython("__import__('types').ModuleType('__main__').__dict__");
        const s = performance.now();
        await py.runPythonAsync(code, { globals: g });
        times.push(performance.now() - s);
        g.destroy();
        if (first === null) first = out;
    }
    times.sort((a, b) => a - b);
    steps[name] = { runMs: times.map((x) => Math.round(x)), medianMs: Math.round(times[1]), stdout: first };
}
const rec = {
    runId: 'lltAttention-t0-pyodide', module: 'lltAttention', tier: 'T0', label: 'baseline',
    hardware: { gpu: null, count: 0, vramGB: 0, cpu: os.cpus()[0]?.model, threads: os.cpus().length, os: `${os.platform()} ${os.release()}` },
    env: { pyodide: py.version, node: process.version },
    notebookCommit: null, config: { code: 'cells/step1.py to step3.py and challenge_solution.py, 3 runs each, fresh namespace per run' }, seed: 0,
    metrics: [], evals: { pyodideLoadMs: loadMs, steps }, samples: [],
    wallClockSec: Math.round((Date.now() - t0) / 10) / 100, costUSD: 0, priceCheckedOn: null,
    date: new Date().toISOString().slice(0, 10),
};
fs.mkdirSync(runsDir, { recursive: true });
fs.writeFileSync(path.join(runsDir, 'lltAttention-t0-pyodide.json'), JSON.stringify(rec, null, 1));
console.log(JSON.stringify(rec.evals, null, 1));
