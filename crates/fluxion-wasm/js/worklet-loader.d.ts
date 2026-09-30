// Types for `fluxion/worklet` — worklet-loader.mjs, and the messages its node's port carries.

/** Options for {@link attachWorklet}. */
export interface WorkletOptions {
  /**
   * `true` makes an **insert** node: one input, processed quantum by quantum as the Web Audio
   * graph delivers it, with zero added latency. Otherwise the node has no input and plays audio
   * posted to it (`{ type: "audio" }`) through a ring. Default `false`.
   */
  input?: boolean;
  /**
   * Independent channels, each with its own chain state — the batch engine's per-channel
   * semantics. An insert node takes exactly this many in (Web Audio up/down-mixes whatever is
   * connected, by the "speakers" rules) and puts this many out. Default 1.
   */
  channels?: number;
  /** Ring mode: frames of audio the ring can hold per channel. Default 8192. */
  ringFrames?: number;
  /**
   * Ring mode: output channels, when they should differ from `channels`; extra ones repeat the
   * last channel, so a mono chain can fill a stereo output. Ignored by insert nodes.
   */
  outputChannels?: number;
  /** Where the package's files are served from. Default: beside worklet-loader.mjs. */
  base?: string | URL;
  /**
   * Ring mode: audio placed in the ring before the first block — the only way to feed an
   * OfflineAudioContext, which can finish rendering before a message arrives. Same shape as
   * {@link AudioMessage.samples}.
   */
  prime?: Float32Array | Float32Array[];
}

/**
 * Register the processor and return a node running `chain` at the context's sample rate.
 */
export function attachWorklet(
  context: BaseAudioContext,
  chain: string,
  options?: WorkletOptions,
): Promise<AudioWorkletNode>;

/** Ask the node how it is doing; resolves with its {@link WorkletStats}. */
export function stats(node: AudioWorkletNode): Promise<WorkletStats>;

// --- page -> worklet ---------------------------------------------------------------------------

/** Ring mode: audio to play. One array for every channel, or one per channel. */
export interface AudioMessage {
  type: "audio";
  samples: Float32Array | Float32Array[];
}

/** Ramp the output gain (linear) to `value` over `rampMs` (default 20). */
export interface GainMessage {
  type: "gain";
  value: number;
  rampMs?: number;
}

/** Clear the chain's state (filter memory, delay lines, tails) and the ring — for stop or seek. */
export interface ResetMessage {
  type: "reset";
}

/** Ask for a {@link WorkletStats} reply. */
export interface StatsRequest {
  type: "stats";
}

/** Stop processing for good; the node can then be garbage collected. */
export interface StopMessage {
  type: "stop";
}

/** Everything a page can post to the node's port. */
export type ToWorklet =
  | AudioMessage
  | GainMessage
  | ResetMessage
  | StatsRequest
  | StopMessage;

// --- worklet -> page ---------------------------------------------------------------------------

/** The reply to {@link StatsRequest}. */
export interface WorkletStats {
  type: "stats";
  /** Quanta processed since the node was built or last reset. */
  rendered: number;
  /** Ring mode: quanta the ring could not fill (played as silence). Always 0 for an insert node. */
  dropped: number;
  /** Ring mode: frames waiting in the ring. Always 0 for an insert node. */
  buffered: number;
  /** Allocations the wasm module has made since it loaded — flat while playing. */
  allocations: number;
}

/** Everything the node's port can post back. */
export type FromWorklet = WorkletStats;
