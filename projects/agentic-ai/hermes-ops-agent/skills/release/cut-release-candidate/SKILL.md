---
name: cut-release-candidate
description: Cut and publish an RC tag for the api service, including the changelog and the two checks CI does not run
version: 1.0.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [release, git, ci]
    category: release
---

## When to Use

The user asks to cut an RC, prepare a release candidate, or tag a pre-release
for the api service. NOT for final releases, which go through
`promote-rc-to-release`.

Level 0 skill matching runs against the frontmatter `description` and this
section, so keep both concrete. A vague description means the skill sits on
disk and never loads.

## Procedure

1. Confirm main is green. Do not proceed on a failing or in-progress run.

       gh run list --branch main --limit 1

2. Find the current RC number.

       git tag --list 'v*-rc*' --sort=-v:refname | head -1

3. Generate the changelog from the last RELEASE tag, not the last RC.

       git log $(git describe --tags --abbrev=0 --match 'v*' --exclude '*-rc*')..HEAD --oneline

4. Lint migrations by hand. CI does not do this on tag builds.

       make lint-migrations

5. Tag and push the one tag by name.

       git tag -a vX.Y.Z-rcN -m "RCN"
       git push origin vX.Y.Z-rcN

## Pitfalls

- Step 3 against the last RC tag produces an empty changelog and the release
  ships with no notes. Nothing errors, so this is only caught by looking.
- `git push --tags` pushes every local tag, including experiments. Push the
  one tag by name.
- Migrations are not linted on tag builds. Step 4 is not optional.

## Verification

    gh release view vX.Y.Z-rcN

Shows the tag, marked pre-release, with a non-empty body. An empty body means
step 3 used the wrong base tag.
