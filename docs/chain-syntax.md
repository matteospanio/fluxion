# The chain syntax

One string describes a chain, and every interface reads it the same way:

```bash
fluxion --chain "highpass(80, 4) | gain(-3dB)" in.wav out.wav   # CLI
```
```python
fx.chain("highpass(80, 4) | gain(-3dB)")                        # Python
```
```c
fx_chain_from_text("highpass(80, 4) | gain(-3dB)");             /* C */
```
```js
Chain.fromText("highpass(80, 4) | gain(-3dB)");                 // JS
```
```rust
"highpass(80, 4) | gain(-3dB)".parse::<Graph>()?;               // Rust
```

It is also what Fluxion **prints**: the parser is the exact inverse of `Graph`'s `Display`, so a
chain read back from `--dry-run`, a `.fxg` dump or `str(chain)` in Python is a chain you can paste
straight back in. That property is asserted in
[`crates/fluxion-core/tests/parse_roundtrip.rs`](../crates/fluxion-core/tests/parse_roundtrip.rs).

## Operators

| Written | Means |
|---|---|
| `a \| b` | **series** — run `a`, feed its output to `b` |
| `a + b` | **parallel** — run both on the same input, sum the outputs |
| `a ~ b` | **feedback** — `y[n] = a(x[n] + b(y)[n-1])` |
| `name: a` | **label** — names the node so it can be addressed later; changes nothing about what it does |
| `id` | pass-through |

`+` binds tighter than `|`, the same way it does in Rust and Python — so
`a | b + c | d` means `a | (b + c) | d` and needs no parentheses. `~` binds loosest, and does not
chain: write `(a ~ b) ~ c` if that is what you mean. A label binds tightest of all, so
`lp: gain(1) | gain(2)` labels only the first node.

## Ops and their parameters

An op is its name, optionally followed by parameters:

```text
lowpass                       every parameter takes its default
lowpass(800)                  the rest still take their defaults
lowpass(800, 4)               positional
lowpass(cutoff=800, order=4)  named
lowpass=800                   shorthand, same as lowpass(800)
fir=0.5,0.3,0.2               the shorthand is how a variadic op reads best
```

Positional values fill left to right. Once a named value appears every later one must be named too.
Anything you leave out takes its default — see [ops.md](ops.md) for every op's parameters, defaults
and ranges.

## Numbers

Plain decimals, with an optional exponent: `800`, `0.707`, `-24`, `1e-3`, `inf`.

Two suffixes:

| Suffix | Means | Example |
|---|---|---|
| `k` | ×1000 | `lowpass=4.41k` is 4410 Hz |
| `dB` | the linear factor `10^(x/20)` | `gain(-3dB)` is 0.708 |

`dB` exists because `gain`'s parameter is a **linear ratio**, so `gain(-3)` is a 3× boost with the
phase inverted — almost certainly not what someone typing `-3` meant. On a parameter that is
already in decibels (`overdrive`'s drive, `peaking`'s gain) the suffix simply restates the unit. On
a parameter that is neither — a frequency, a time — it is an error rather than a silent
misreading.

Suffixes are input only. Rendering never emits them, so everything comes back in one canonical
form.

## Errors

Errors carry the position, the problem and, where there is one, the fix:

```text
error: unknown op 'hipass'
  hipass=80 | gain=-3dB
  ^^^^^^ did you mean 'highpass'?
```

The suggestion counts a transposition as a single edit, so `gian` finds `gain`. When nothing is
close enough to be a plausible typo you get no suggestion — a wrong guess is worse than none.

Out-of-range values point at the value, not at the op:

```text
error: op 'lowpass' parameter 'cutoff' = -5 is out of range [0, inf]
  lowpass(-5)
          ^^
```

## Grammar

Loosest binding to tightest:

```ebnf
chain    = keyed ;
keyed    = feedback [ "<" feedback ] ;      (* non-associative *)
feedback = series [ "~" series ] ;          (* non-associative *)
series   = parallel { "|" parallel } ;      (* left-associative *)
parallel = labeled { "+" labeled } ;        (* left-associative *)
labeled  = ident ":" labeled | primary ;    (* the label binds tightest *)
primary  = "id" | side | op | "(" chain ")" ;
side     = "side" "(" digits ")" ;          (* a second input, numbered from 0 *)
op       = ident [ "(" [ args ] ")" | "=" values ] ;
args     = arg { "," arg } ;
arg      = number | ident "=" number ;      (* positional first, then named *)
values   = number { "," number } ;
number   = [ "-" ] ( digits [ "." digits ] [ exp ] | "inf" ) [ suffix ] ;
suffix   = "k" | "dB" ;                     (* case-insensitive *)
ident    = ( alpha | "_" ) { alpha | digit | "_" } ;
```

There is no subtraction operator, so a leading `-` always starts a number; conversely `+` is always
the parallel operator, so numbers are never written with a leading `+`.

## Side inputs and keys

Most chains carry one signal. Some need two: a gate on a drum overhead opened by the snare's own
microphone, a bass ducked by a kick. `side(0)` reads the first extra signal handed to the chain
instead of what is flowing down it, and `<` says which signal drives a keyed op:

```
gate(-35, 40) < side(0)                  # gated by the first side input
gate(-35, 40) < side(0) | lowpass(200)   # ...listening only to its low end
lowpass(800) + side(0)                   # not a key at all: just mixing a second signal in
```

`<` is the loosest operator and does not chain: bracket what you mean, as with `~`. The key runs on
the same input the node was given, so the low-pass above filters the *key*, not the programme.

Only ops that declare a key input read it — `gate` is the one today — so keying a chain of ordinary
ops changes nothing. A `side(n)` with no signal connected reads as silence, which is what lets the
same chain text be handed to an interface that has no way to pass a second signal. Note what that
means for a gate: `gate(...) < side(0)` run without a side signal hears silence and *closes*. That
is deliberate — a key that went missing should shut the gate, not quietly fall back to opening it.

Side signals are supplied per interface: `--side file.wav` on the CLI, `process_with` in Rust,
`sides=[...]` in Python, `processWith` in the browser.

The implementation is [`crates/fluxion-core/src/parse.rs`](../crates/fluxion-core/src/parse.rs).
