# 0007. One VCS seam, so the demo and GitHub are the same code path

**Status:** Accepted

## Context

The demo has to work offline against fixture repos. The real thing has to open pull
requests on GitHub. If those are two code paths, the demo proves nothing about the
real one, and the first live run is the first time the integration is exercised.

## Options

1. **Two paths with a branch at the top.** Simplest to write, and the demo stops
   being evidence about production.
2. **Always talk to GitHub**, using a scratch org for the demo. Faithful, but
   requires credentials and a network to run anything at all.
3. **A provider interface** with a local-git implementation and a GitHub one.

## Decision

Option 3. `vcs/base.py` defines the interface (`checkout`, `commit`, `open_pr`);
`vcs/local.py` implements it against bare git repositories on disk, `vcs/github.py`
against the API, and `vcs/factory.py` selects by configuration. The orchestrator
knows only the interface. "Opening a PR" locally means pushing a branch to a bare
repo and recording the request - the same sequence, minus the network.

## Consequences

**Good**

- The demo exercises the real checkout/commit/branch sequence, so what is proven
  offline is most of what happens online.
- Adding GitLab or Bitbucket is a new implementation, not a change to the
  orchestrator.
- Tests can run the whole fleet against local bare repos with no credentials.

**Bad**

- **The GitHub implementation is the least-tested code in the project** (~31% line
  coverage) precisely *because* the seam makes it easy to avoid. The abstraction
  that lets the demo run offline is the same abstraction that lets the real
  provider go unexercised, and the failures that matter in production - rate
  limits, permissions, merge conflicts, branch protection - all live on that side.
  Treat that number as a standing warning, not a to-do someone will get to.
- The interface is shaped by what local git can do. Anything GitHub offers that
  has no local analogue (review requests, required checks, draft PRs) either sits
  outside the interface or gets a no-op local implementation.
- Two implementations of "open a PR" can drift in meaning while both satisfying the
  signature.
