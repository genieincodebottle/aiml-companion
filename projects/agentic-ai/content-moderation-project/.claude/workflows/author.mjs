// Generate-then-verify pipeline: scout (cheap) -> author (top) -> verify (mid, parallel).
// Only the author stage runs on the top model. Scout and verify are cheap.
// Run this with Claude Code's Workflow tool, or adapt the shape to your own runner.
//
// Pass the task in as args, e.g. { task: "add a rate limit to the appeal endpoint" }.

export const meta = {
    name: 'author',
    description: 'Scout the files, author the change on the top model, verify on mid models in parallel.',
    phases: [
        { title: 'Scout' },
        { title: 'Author' },
        { title: 'Verify' },
    ],
};

const task = (args && args.task) || 'see the prompt';

// 1. SCOUT (cheap): find the files and patterns to reuse. Read-only.
phase('Scout');
const scout = await agent(
    `Find every file involved in this task and the patterns to reuse: ${task}`,
    { label: 'scout', model: 'haiku' }
);

// 2. AUTHOR (top): the only expensive stage.
phase('Author');
const change = await agent(
    `Using this scouting report, make the change. Follow the repo's patterns and CLAUDE.md.\n\n${scout}`,
    { label: 'author', model: 'opus' }
);

// 3. VERIFY (mid, in parallel): independent reviewers, each told to refute through a different lens.
phase('Verify');
const lenses = ['correctness', 'edge cases', 'does it match repo patterns'];
const verdicts = await parallel(
    lenses.map((lens) => () =>
        agent(
            `Review this change through the "${lens}" lens. Try to refute it. ` +
            `Report issues as file:line plus one sentence, then ship / do-not-ship.\n\n${change}`,
            { label: `verify:${lens}`, model: 'sonnet' }
        )
    )
);

return { change, verdicts };