/**
 * Galamsey AI Detector — microphone capture processor
 * ===================================================
 *
 * Runs on the browser's audio rendering thread (AudioWorklet), loaded
 * by sound-detection.js. It does one thing: copy the raw microphone
 * samples out to the page in batches.
 *
 * The audio thread delivers 128 samples at a time (~375 posts/second
 * at 48 kHz). Batching to 4096 keeps the message traffic to the page
 * at ~12/second. Nothing is dropped: every sample ends up in exactly
 * one batch, so the page can cut windows with no gaps between them.
 *
 * The microphone is requested as mono, but browsers may still deliver
 * two channels. They are averaged to mono here, the same mix-down the
 * API applies (audio.mean(axis=1) in src/preprocessing.py).
 */

const BATCH_SAMPLES = 4096;

class SoundCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.batch = new Float32Array(BATCH_SAMPLES);
        this.length = 0;
    }

    process(inputs) {
        const channel = mixToMono(inputs[0]);
        if (channel) {
            let offset = 0;
            while (offset < channel.length) {
                const count = Math.min(channel.length - offset, BATCH_SAMPLES - this.length);
                this.batch.set(channel.subarray(offset, offset + count), this.length);
                this.length += count;
                offset += count;
                if (this.length === BATCH_SAMPLES) {
                    // Hand the buffer over (no copy) and start a fresh one.
                    this.port.postMessage(this.batch, [this.batch.buffer]);
                    this.batch = new Float32Array(BATCH_SAMPLES);
                    this.length = 0;
                }
            }
        }
        // Outputs are left silent, so nothing is played back.
        return true;
    }
}

function mixToMono(channels) {
    if (!channels || channels.length === 0) return null;
    if (channels.length === 1) return channels[0];
    const mono = new Float32Array(channels[0].length);
    for (const channel of channels) {
        for (let i = 0; i < mono.length; i++) mono[i] += channel[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= channels.length;
    return mono;
}

registerProcessor('sound-capture', SoundCaptureProcessor);
