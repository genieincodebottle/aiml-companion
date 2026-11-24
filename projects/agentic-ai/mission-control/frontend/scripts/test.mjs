// Runs Vitest from paths with an upper-case drive letter.
//
// Windows only: a terminal opened in `c:\...` (lower case, common in VS Code and
// Git Bash) makes Vitest load itself under two path spellings and fail with
// "Vitest failed to find the runner". On macOS and Linux this changes nothing.
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const cwd = process.cwd().replace(/^[a-z]:/, (drive) => drive.toUpperCase());
const vitest = join(cwd, 'node_modules', 'vitest', 'vitest.mjs');
const result = spawnSync(process.execPath, [vitest, 'run', ...process.argv.slice(2)], { cwd, stdio: 'inherit' });
process.exit(result.status ?? 1);
