# Fluxion documentation

- **[interfaces.md](interfaces.md)** — the contract between Fluxion's five doors (Rust, CLI,
  Python, C, JS/wasm): who each is for, the rule that an op ships on all of them or this document
  says why not, and the definition of done a pull request is held to.
- **[chain-syntax.md](chain-syntax.md)** — the one text form for a chain, shared by every
  interface. Operators, parameters, units, errors, grammar.
- **[ops.md](ops.md)** — *generated.* Every op, what it does, its parameters and ranges, and its
  name on each interface. Regenerate with `python scripts/gen_interfaces.py`; CI fails on a hand
  edit.

Architecture and the paper are in the [README](../README.md); conventions are in
[CONTRIBUTING.md](../CONTRIBUTING.md); what is planned and in what order is in
[ROADMAP.md](../ROADMAP.md).
