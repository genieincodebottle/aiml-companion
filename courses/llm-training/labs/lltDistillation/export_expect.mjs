// Write expect.json: the page's "What you should see" text per step, for build_notebook.py.
//   node export_expect.mjs <assembled module .mjs>
import { writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const { lltDistillation: m } = await import(pathToFileURL(process.argv[2]).href);
const out = {};
m.lesson.build.forEach((s, i) => { out[String(i + 1)] = s.expect; });
writeFileSync(new URL('./expect.json', import.meta.url), JSON.stringify(out, null, 1));
console.log('steps', Object.keys(out).length);