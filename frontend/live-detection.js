/**
 * Galamsey AI Detector — live camera inference
 * ============================================
 *
 * The missing link. This file is what turns the dashboard from "a page
 * that shows your webcam" into "a page that detects things".
 *
 *      <video>  (browser renders the camera)
 *          |
 *          v  drawImage once every DETECTION_INTERVAL_MS
 *      <canvas>  (offscreen, never displayed)
 *          |
 *          v  toBlob -> JPEG
 *      POST /api/detect
 *          |
 *          v  real YOLOv12 inference on galamsey-phase5.pt
 *      JSON detections
 *          |
 *          +--> bounding-box overlay   (a separate layer ON TOP of the video)
 *          +--> Detection Report
 *          +--> statistics
 *          +--> alerts
 *
 * --------------------------------------------------------------
 * WHY THE OVERLAY IS A SEPARATE LAYER
 * --------------------------------------------------------------
 * The backend could draw boxes onto the frame and send back a picture.
 * We deliberately don't:
 *
 *   * The browser already renders the camera at 30-60 fps for free,
 *     with hardware acceleration. Round-tripping every frame through
 *     Python to draw rectangles would make the video jerky and pointless.
 *   * Boxes burned into pixels cannot be restyled, hidden, hovered,
 *     clicked or made accessible.
 *   * Video and detections update at different rates. The video is
 *     smooth; detections arrive ~every 1.5 s. Separate layers let each
 *     run at its natural rate.
 *
 * So: the browser owns the picture, the backend owns the inference, and
 * absolutely-positioned <div>s float on top.
 *
 * --------------------------------------------------------------
 * WHY ONLY ONE REQUEST IS EVER IN FLIGHT
 * --------------------------------------------------------------
 * A naive `setInterval(sendFrame, 100)` posts a frame every 100 ms
 * regardless of whether the previous one came back. Inference takes
 * ~110 ms, so requests pile up faster than they drain: the queue grows
 * without limit, memory climbs, and the boxes you see get further and
 * further behind the picture.
 *
 * This loop is therefore capture -> infer -> render -> WAIT -> repeat,
 * with an explicit in-flight guard. If a cycle takes longer than the
 * interval, the next one simply starts later. The system degrades by
 * slowing down, which is what you want, instead of by falling over.
 *
 * --------------------------------------------------------------
 * THE FREQUENCY TRADE-OFF
 * --------------------------------------------------------------
 *   faster (e.g. 200 ms)  more responsive boxes, but the backend is
 *                         busy ~55% of the time, the laptop heats up,
 *                         and ~5 JPEGs/second cross the network
 *   slower (e.g. 5 s)     almost free, but boxes lag reality badly
 *   1500 ms (default)     backend ~93% idle, boxes feel current for
 *                         scenes that change at human speed
 *
 * Mining sites do not change in 200 ms. 1.5 s is the honest choice here.
 * Configurable via GALAMSEY_CONFIG.DETECTION_INTERVAL_MS.
 */

(function () {
    'use strict';

    class LiveDetection {
        /**
         * @param {object} dashboard the GalamseyDetector instance from script.js
         */
        constructor(dashboard) {
            this.dashboard = dashboard;
            this.running = false;
            this.inFlight = false;          // the guard that prevents queueing
            this.timer = null;
            this.lastResult = null;
            this.consecutiveErrors = 0;

            // Session totals, derived ONLY from real API responses.
            this.stats = {
                frames: 0,
                inferences: 0,
                totalDetections: 0,
                byClass: {},
                lastInferenceMs: null
            };

            this.canvas = document.createElement('canvas');   // offscreen
            this.ctx = this.canvas.getContext('2d');

            // Keep the overlay aligned when the window (and therefore the
            // video element) changes size. Boxes are positioned in pixels,
            // so they must be recomputed, not just left where they were.
            this._onResize = () => this.renderBoxes(this.lastResult);
            window.addEventListener('resize', this._onResize);
        }

        get config() {
            return window.GALAMSEY_CONFIG || {};
        }

        get intervalMs() {
            return this.config.DETECTION_INTERVAL_MS || 1500;
        }

        // -----------------------------------------------------------
        // Lifecycle
        // -----------------------------------------------------------
        start() {
            if (this.running) return;
            this.running = true;
            this.consecutiveErrors = 0;
            console.log(`[galamsey] live detection started (every ${this.intervalMs} ms)`);
            this.dashboard.showToast('Live detection started', 'legal');
            this.scheduleNext(0);
        }

        stop() {
            this.running = false;
            if (this.timer) {
                clearTimeout(this.timer);
                this.timer = null;
            }
            this.clearBoxes();
            console.log('[galamsey] live detection stopped');
        }

        scheduleNext(delay) {
            if (!this.running) return;
            this.timer = setTimeout(() => this.tick(), delay === undefined ? this.intervalMs : delay);
        }

        /**
         * One full cycle. Always reschedules, even on failure, so a
         * transient backend hiccup does not kill the loop permanently.
         */
        async tick() {
            if (!this.running) return;

            if (this.inFlight) {
                // Should not happen given the flow below, but if it ever
                // does we skip rather than pile up.
                this.scheduleNext();
                return;
            }

            this.inFlight = true;
            try {
                const blob = await this.captureFrame();
                if (blob) {
                    const result = await this.sendForInference(blob);
                    if (result && result.success) {
                        this.consecutiveErrors = 0;
                        this.handleResult(result);
                    }
                }
            } catch (error) {
                this.consecutiveErrors += 1;
                if (this.consecutiveErrors <= 3) {
                    console.warn('[galamsey] detection cycle failed:', error.message);
                }
                if (this.consecutiveErrors === 3) {
                    this.dashboard.showToast(
                        'Detection backend unreachable - retrying in the background',
                        'illegal'
                    );
                }
            } finally {
                this.inFlight = false;
                // Back off when the backend is unhappy instead of hammering it.
                const backoff = this.consecutiveErrors > 3 ? this.intervalMs * 4 : undefined;
                this.scheduleNext(backoff);
            }
        }

        // -----------------------------------------------------------
        // Capture
        // -----------------------------------------------------------
        /**
         * Grab the current video frame as a JPEG Blob.
         *
         * We downscale to DETECTION_MAX_WIDTH before sending. The model
         * resizes everything to 640x640 internally anyway, so shipping a
         * 1920px frame wastes bandwidth and encode time for no accuracy.
         * Aspect ratio is preserved, which is what lets the returned
         * coordinates still map correctly onto the displayed video.
         */
        captureFrame() {
            const video = document.getElementById('webcamVideo');
            if (!video || !video.videoWidth || video.readyState < 2) return Promise.resolve(null);

            const maxWidth = this.config.DETECTION_MAX_WIDTH || 960;
            const scale = Math.min(1, maxWidth / video.videoWidth);
            this.canvas.width = Math.round(video.videoWidth * scale);
            this.canvas.height = Math.round(video.videoHeight * scale);
            this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
            this.stats.frames += 1;

            return new Promise((resolve) => {
                this.canvas.toBlob(
                    resolve,
                    'image/jpeg',
                    this.config.DETECTION_JPEG_QUALITY || 0.8
                );
            });
        }

        async sendForInference(blob) {
            const form = new FormData();
            form.append('image', blob, 'frame.jpg');

            const response = await fetch(`${this.config.BACKEND_URL}/api/detect`, {
                method: 'POST',
                body: form
            });

            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                throw new Error(detail.error || `HTTP ${response.status}`);
            }
            return response.json();
        }

        // -----------------------------------------------------------
        // Results
        // -----------------------------------------------------------
        handleResult(result) {
            // Refuse to display synthetic data as if it were real.
            if (result.synthetic) {
                console.error('[galamsey] backend is in MOCK mode - not displaying as real results');
                this.dashboard.showToast(
                    'Backend is in MOCK mode: these are NOT real AI detections',
                    'illegal'
                );
            }

            this.lastResult = result;
            this.stats.inferences += 1;
            this.stats.lastInferenceMs = result.inference_time_ms;
            this.stats.totalDetections += result.detections.length;
            result.detections.forEach((d) => {
                this.stats.byClass[d.class_name] = (this.stats.byClass[d.class_name] || 0) + 1;
            });

            this.renderBoxes(result);
            this.updateReport(result);
            this.updateVideoStats(result);
            this.raiseAlerts(result);
        }

        // -----------------------------------------------------------
        // Overlay
        // -----------------------------------------------------------
        /**
         * Map a box from SOURCE-IMAGE pixels to DISPLAYED pixels.
         *
         * This is the part people get wrong. Three coordinate spaces are
         * involved and they are all different:
         *
         *   1. the frame we POSTed        e.g.  960 x 540
         *   2. the <video> element's box  e.g. 1005 x 700
         *   3. the picture as actually painted inside that box
         *
         * The video is styled `object-fit: cover`, which scales the
         * picture until it FILLS the box and crops whatever overflows -
         * it does not stretch. So the visible picture is generally BIGGER
         * than the element and is centred, with parts hidden off both
         * sides (or top and bottom).
         *
         * Naive scaling (x * elementWidth / sourceWidth) ignores the crop
         * and puts every box in the wrong place, subtly, in a way that
         * looks almost right. The correct transform is:
         *
         *   scale   = max(boxW / srcW, boxH / srcH)     <- max, not min
         *   offsetX = (boxW - srcW * scale) / 2         <- negative when cropped
         *   screenX = srcX * scale + offsetX
         */
        mapToDisplay(video, srcW, srcH) {
            const boxW = video.clientWidth;
            const boxH = video.clientHeight;
            const scale = Math.max(boxW / srcW, boxH / srcH);   // "cover"
            return {
                scale,
                offsetX: (boxW - srcW * scale) / 2,
                offsetY: (boxH - srcH * scale) / 2
            };
        }

        clearBoxes() {
            const overlay = document.querySelector('.detection-overlay');
            if (overlay) overlay.querySelectorAll('.live-box').forEach((el) => el.remove());
        }

        renderBoxes(result) {
            const overlay = document.querySelector('.detection-overlay');
            const video = document.getElementById('webcamVideo');
            if (!overlay || !video || !result) return;

            this.clearBoxes();
            overlay.style.display = 'block';

            const { scale, offsetX, offsetY } =
                this.mapToDisplay(video, result.image_width, result.image_height);

            result.detections.forEach((d) => {
                const b = d.bbox;
                const left = b.x1 * scale + offsetX;
                const top = b.y1 * scale + offsetY;
                const width = (b.x2 - b.x1) * scale;
                const height = (b.y2 - b.y1) * scale;

                // Skip boxes cropped entirely out of view by object-fit: cover.
                if (left + width < 0 || top + height < 0 ||
                    left > video.clientWidth || top > video.clientHeight) return;

                const el = document.createElement('div');
                el.className = `live-box live-box--c${d.class_id}`;
                el.style.left = `${left}px`;
                el.style.top = `${top}px`;
                el.style.width = `${width}px`;
                el.style.height = `${height}px`;

                const label = document.createElement('span');
                label.className = 'live-box__label';
                label.textContent =
                    `${this.displayName(d.class_id, d.class_name)} ${(d.confidence * 100).toFixed(1)}%`;
                el.appendChild(label);
                overlay.appendChild(el);
            });
        }

        displayName(classId, fallback) {
            const known = (this.config.CLASSES || []).find((c) => c.id === classId);
            return known ? known.display : fallback;
        }

        // -----------------------------------------------------------
        // Dashboard
        // -----------------------------------------------------------
        /**
         * One inference result drives every panel. The overlay, the
         * report and the statistics all read from the SAME response, so
         * they can never disagree about what was seen.
         */
        updateReport(result) {
            const dets = result.detections;

            // Detected classes, with a count and the best confidence each.
            const byClass = new Map();
            dets.forEach((d) => {
                const entry = byClass.get(d.class_id) || { count: 0, best: 0, name: d.class_name };
                entry.count += 1;
                entry.best = Math.max(entry.best, d.confidence);
                byClass.set(d.class_id, entry);
            });

            const toolList = document.getElementById('toolList');
            if (toolList) {
                toolList.innerHTML = '';
                if (byClass.size === 0) {
                    const empty = document.createElement('div');
                    empty.className = 'tool-item';
                    empty.innerHTML = '<span>No objects detected</span>';
                    toolList.appendChild(empty);
                } else {
                    [...byClass.entries()]
                        .sort((a, b) => b[1].count - a[1].count)
                        .forEach(([classId, info]) => {
                            const item = document.createElement('div');
                            item.className = 'tool-item';
                            item.innerHTML =
                                `<i class="fas fa-crosshairs"></i>` +
                                `<span>${this.displayName(classId, info.name)} ×${info.count}</span>` +
                                `<span class="confidence">${(info.best * 100).toFixed(0)}%</span>`;
                            toolList.appendChild(item);
                        });
                }
            }

            // Mean confidence across the detections actually returned.
            const confidenceEl = document.getElementById('confidenceValue');
            if (confidenceEl) {
                const mean = dets.length
                    ? dets.reduce((sum, d) => sum + d.confidence, 0) / dets.length
                    : 0;
                confidenceEl.textContent = `${(mean * 100).toFixed(1)}%`;
            }

            // "Site type" reports what the MODEL SAW. It does NOT claim
            // legality: that needs geofence context and a human. See
            // backend/taxonomy.py MINING_ACTIVITY_WEIGHT.
            const siteType = document.getElementById('siteType');
            if (siteType) {
                if (dets.length === 0) {
                    siteType.textContent = 'No objects detected';
                    siteType.className = 'site-type';
                } else {
                    const score = this.miningActivityScore(byClass);
                    siteType.className = 'site-type';
                    siteType.textContent = score >= 1.0
                        ? 'Mining equipment present'
                        : score > 0
                            ? 'Activity detected'
                            : 'Objects detected';
                }
            }

            const environment = document.getElementById('environment');
            if (environment) {
                environment.textContent = dets.length
                    ? `${dets.length} object${dets.length === 1 ? '' : 's'} in frame`
                    : 'Monitoring - nothing detected';
            }

            const action = document.getElementById('suggestedAction');
            if (action) {
                action.textContent = this.suggestedAction(byClass, dets.length);
            }
        }

        /**
         * How strongly the detected objects indicate industrial mining.
         * Mirrors backend/taxonomy.py MINING_ACTIVITY_WEIGHT.
         *
         * This is NOT a legality judgement. A Person alone scores low on
         * purpose: people are not evidence of a crime.
         */
        miningActivityScore(byClass) {
            const weights = { 0: 1.0, 1: 0.8, 2: 0.2, 3: 1.0, 4: 0.4, 5: 0.3 };
            let max = 0;
            byClass.forEach((_info, classId) => {
                max = Math.max(max, weights[classId] || 0);
            });
            return max;
        }

        suggestedAction(byClass, total) {
            if (total === 0) return 'Monitoring - no action required';
            const heavy = byClass.has(0) || byClass.has(3);   // machinery or processing rig
            if (heavy) {
                return 'Mining equipment detected. Check the site location against ' +
                       'permitted concessions and protected areas before escalating.';
            }
            return 'Objects detected. Continue monitoring; no mining equipment identified.';
        }

        updateVideoStats(result) {
            const counter = document.querySelector('.detection-count');
            if (counter) {
                counter.textContent =
                    `${result.detections.length} object${result.detections.length === 1 ? '' : 's'} detected`;
            }
            // NOTE: .frame-counter is owned by script.js's startFrameCounter(),
            // which updates it on its own interval. Writing to it from here too
            // meant the two fought and the value flickered between them. The
            // inference timing is surfaced in the console and via /health
            // instead - one element, one owner.
        }

        /**
         * Alerts fire on OBJECT DETECTION only. Turning a detection into
         * an "illegal mining" claim requires location and geofence
         * context, which the model cannot supply from pixels.
         */
        raiseAlerts(result) {
            const heavy = result.detections.filter(
                (d) => (d.class_id === 0 || d.class_id === 3) && d.confidence >= 0.5
            );
            if (heavy.length === 0) return;

            const signature = heavy.map((d) => d.class_id).sort().join(',');
            if (signature === this._lastAlertSignature) return;    // don't spam
            this._lastAlertSignature = signature;

            const names = [...new Set(heavy.map((d) => this.displayName(d.class_id, d.class_name)))];
            this.dashboard.addAlert(
                `${names.join(' and ')} detected (${heavy.length} object${heavy.length === 1 ? '' : 's'})`,
                'illegal'
            );
        }

        destroy() {
            this.stop();
            window.removeEventListener('resize', this._onResize);
        }
    }

    window.GalamseyLiveDetection = LiveDetection;
})();
