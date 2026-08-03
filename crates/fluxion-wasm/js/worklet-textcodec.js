// A minimal UTF-8 TextEncoder/TextDecoder for AudioWorkletGlobalScope.
//
// `AudioWorkletGlobalScope` is deliberately tiny: no `window`, no `self`, no `fetch` — and no
// `TextDecoder` or `TextEncoder`. wasm-bindgen's glue constructs both at the top of the file, so
// without this the worklet script throws `TextDecoder is not defined` before it reaches
// `registerProcessor`, and the only symptom is an `AudioWorkletNode` that cannot be constructed
// because the processor "is not defined". This is prepended to the worklet module for that reason.
//
// Only what the glue actually uses is implemented. In particular `encodeInto` is deliberately
// absent: wasm-bindgen feature-detects it and takes a plain `encode` path when it is missing, and
// that path is far easier to be sure of than reproducing `encodeInto`'s read/written contract for
// a buffer that may be too small.

/* eslint-disable no-undef */
if (typeof TextEncoder === "undefined") {
  globalThis.TextEncoder = class TextEncoder {
    get encoding() {
      return "utf-8";
    }

    encode(input = "") {
      const bytes = [];
      for (let i = 0; i < input.length; i++) {
        let code = input.codePointAt(i);
        if (code > 0xffff) {
          i++; // a surrogate pair is one code point but two UTF-16 units
        }
        if (code < 0x80) {
          bytes.push(code);
        } else if (code < 0x800) {
          bytes.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
        } else if (code < 0x10000) {
          bytes.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
        } else {
          bytes.push(
            0xf0 | (code >> 18),
            0x80 | ((code >> 12) & 0x3f),
            0x80 | ((code >> 6) & 0x3f),
            0x80 | (code & 0x3f),
          );
        }
      }
      return new Uint8Array(bytes);
    }
  };
}

if (typeof TextDecoder === "undefined") {
  globalThis.TextDecoder = class TextDecoder {
    constructor(label = "utf-8") {
      this.encoding = label;
    }

    decode(input) {
      if (input === undefined) {
        return "";
      }
      const bytes = input instanceof Uint8Array ? input : new Uint8Array(input.buffer ?? input);
      let out = "";
      for (let i = 0; i < bytes.length; ) {
        const first = bytes[i++];
        let code;
        if (first < 0x80) {
          code = first;
        } else if (first < 0xe0) {
          code = ((first & 0x1f) << 6) | (bytes[i++] & 0x3f);
        } else if (first < 0xf0) {
          code = ((first & 0x0f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
        } else {
          code =
            ((first & 0x07) << 18) |
            ((bytes[i++] & 0x3f) << 12) |
            ((bytes[i++] & 0x3f) << 6) |
            (bytes[i++] & 0x3f);
        }
        out += String.fromCodePoint(code);
      }
      return out;
    }
  };
}
