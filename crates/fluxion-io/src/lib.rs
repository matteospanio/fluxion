//! `fluxion-io` — audio and batch IO.
//!
//! Pure-Rust audio decode/encode via Symphonia + hound (no libsndfile/ffmpeg/PortAudio C
//! dependency — the self-containment payoff of `PROJECT.md` §7), plus Apache Arrow / Parquet for
//! the CLI's columnar batch and dataset IO (decode → record batches → Parquet/IPC).
//!
//! Empty scaffold for now.
