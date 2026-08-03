# The Python quickstart. CI runs this from a fresh virtualenv against the built wheel, in a
# temporary working directory, so it makes its own input rather than needing a fixture.
#
# Ten code lines is the budget (docs/interfaces.md). Blank and comment lines do not count. If this
# ever needs an eleventh line, fix the API — not the quickstart.

import numpy as np

import fluxion as fx

fx.Wave(np.random.default_rng(0).standard_normal((1, 48_000)) * 0.1, 48_000).save("in.wav")

wave = fx.Wave.from_file("in.wav")
out = wave | fx.filter.Highpass(80, order=4) | fx.effect.Gain(fx.db(-3))
out.save("out.wav")

assert out.fs == 48_000 and out.channels() == 1 and len(out) == 48_000
print(f"ok: {out!r}")
