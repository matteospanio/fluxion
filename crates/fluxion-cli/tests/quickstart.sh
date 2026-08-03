#!/bin/sh
# The CLI quickstart. tests/quickstart.rs runs this with $FLUXION pointing at the built binary,
# in a temp directory. Ten code lines is the budget (docs/interfaces.md); comments do not count.
set -e

$FLUXION synth --wave sine --freq 440 --secs 1 in.wav

$FLUXION in.wav highpass --cutoff 80 --order 4 gain --db -3 out.wav
$FLUXION --chain "highpass(80, 4) | gain(-3dB)" in.wav same.wav

$FLUXION stat out.wav
