---
name: Feature request
about: Propose a new capability or API
title: ""
labels: enhancement
assignees: ""
---

**What & why**
The capability you want and the problem it solves. Which of the project goals (differentiable DSP,
CLI, minimal deps, realtime/batch efficiency) does it serve?

**Sketch**
A rough API, CLI surface, or graph example — even pseudocode.

```rust
// e.g. let eq = LoShelf::cutoff(200.0).gain_db(6.0) | Gain::db(-1.0);
```

**Fit with the design**
Anything relevant from [PROJECT.md](../../PROJECT.md) / [AGENTS.md](../../AGENTS.md) (two-engine
split, own-the-backward, explicit coeff design). Note any new dependency it would need.

**Alternatives considered**
Other approaches and why this one.
