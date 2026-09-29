/**
 * Galamsey AI Detector — continuous audio monitoring
 * ==================================================
 *
 * Listens to the microphone while Audio Monitoring is ON and sends each
 * 10-second window to the separate mining-sound-detection API. No
 * inference happens in the browser.
 *
 *      microphone  (getUserMedia)
 *          |
 *          v  raw float samples, via sound-capture-worklet.js
 *      10 s window  (collected back-to-back, no gaps)
 *          |
 *          v  encoded here as 16-bit PCM mono WAV
 *      POST {SOUND_API_URL}/predict      (Python FastAPI, port 8000)
 *          |
 *          v  existing Predictor + best.pt
 *      { machinery, vehicles, human_activity, environment, detected }
 *          |
 *          v
 *      Sound Detection panel
 *
 * --------------------------------------------------------------
 * WHY NOT MediaRecorder
 * --------------------------------------------------------------
 * Chrome's MediaRecorder only produces WebM/Opus (and MP4 in recent
 * versions). The API accepts WAV, FLAC and OGG only, so a recording
 * would have to be decoded and re-encoded before sending. Reading the
 * raw samples directly and writing a WAV header around them is simpler:
 * no codec, no decode step, and no gap between windows, because the
 * recorder never has to be stopped and restarted.
 *
 * The WAV keeps the microphone's native sample rate. The API mixes to
 * mono and resamples to 16 kHz itself (src/preprocessing.py), so the
 * model sees audio prepared exactly as it was during training.
 *
 * --------------------------------------------------------------
 * WHY THE BROWSER'S VOICE PROCESSING IS OFF
 * --------------------------------------------------------------
 * By default the browser applies echo cancellation, noise suppression
 * and automatic gain to the microphone. Those are tuned for speech
 * calls: noise suppression in particular removes steady engine and
 * machinery sound, which is exactly what this model listens for. They
 * are disabled so the model hears the raw scene.
 *
 * --------------------------------------------------------------
 * ONE REQUEST AT A TIME
 * --------------------------------------------------------------
 * Capture never stops, but if a window completes while the previous
 * one is still being analysed, the new window is skipped instead of
 * queued. A slow or stalled API therefore cannot build up a backlog of
 * stale audio (same reasoning as live-detection.js).
 *
 * --------------------------------------------------------------
 * THE FOUR SCORES ARE INDEPENDENT
 * --------------------------------------------------------------
 * The model is multi-label: a window can contain machinery AND vehicles
 * AND people at once. Each score is its own sigmoid output, so they do
 * not add up to 100% and are never normalised. They are shown exactly
 * as returned, and the detected list comes from the API's `detected`
 * array so the 0.50 threshold lives in one place.
 *
 * --------------------------------------------------------------
 * DETECTION EVENTS: ONE STATE PER CATEGORY
 * --------------------------------------------------------------
 * A sound that lasts a minute spans six windows. Alerting on every
 * positive window would alert six times for one thing happening, so
 * windows are grouped into events. Each of the four categories has its
 * own two-state machine, driven only by that category's probability:
 *
 *      IDLE      --(p >= 0.50)-->  DETECTED   new event: ONE alert
 *      DETECTED  --(p >= 0.50)-->  DETECTED   same event: update confidence only
 *      DETECTED  --(p <  0.50)-->  IDLE       event ends, no alert
 *      IDLE      --(p <  0.50)-->  IDLE
 *
 * `this.detections[key]` IS the state: null means IDLE, an object means
 * DETECTED. An alert can only be created on the null -> object
 * transition, which is what stops it repeating. Only the categories
 * ticked under Detection Targets run the machine. Ticking or unticking
 * a category, and turning monitoring on or off, reset the affected
 * states to null, so nothing carries over.
 *
 * A failed request never reaches the state machines: no response is
 * not a zero, and not a detection. Every state is left as it was until
 * the next successful prediction.
 *
 * --------------------------------------------------------------
 * ALERTS PERSIST UNTIL ACKNOWLEDGED
 * --------------------------------------------------------------
 * A new event creates an alert that stays on the panel, and keeps one
 * shared alarm repeating, until the user presses Acknowledge. The
 * alert is separate from the detection: acknowledging silences it but
 * does not end the event, so a sound that is still present stays in
 * Active Detections and cannot re-alert until it has dropped below
 * 0.50 and risen again.
 *
 * However many alerts are open there is only ever one alarm timer, and
 * it plays through the monitoring AudioContext (see playBeep).
 * Turning monitoring off silences every open alert but leaves it on
 * screen to acknowledge; alerts from later events sound as usual.
 *
 * These are sound-category detections, not verdicts about legality:
 * the model hears machinery, vehicles, people and environment only.
 */

(function () {
    'use strict';

    // Order and display names match NAMES in the API's src/api.py.
    const CATEGORIES = [
        { key: 'machinery',      display: 'Machinery',      event: 'Machinery detected' },
        { key: 'vehicles',       display: 'Vehicles',       event: 'Vehicle activity detected' },
        { key: 'human_activity', display: 'Human Activity', event: 'Human activity detected' },
        { key: 'environment',    display: 'Environment',    event: 'Environmental sound detected' }
    ];

    // Matches CLIP_SECONDS in the model's config: one window, one clip.
    const DEFAULT_WINDOW_SECONDS = 10;

    // Same threshold the API uses for `detected` (THRESHOLD in src/api.py).
    const DETECTION_THRESHOLD = 0.50;

    const MAX_RECENT_ALERTS = 10;

    // Alarm pattern: beep, beep, pause - repeated every ALARM_PERIOD_MS.
    const ALARM_PERIOD_MS = 1500;
    const BEEP_OFFSETS = [0, 0.25];     // seconds into each period
    const BEEP_SECONDS = 0.18;

    class SoundDetection {
        constructor() {
            this.toggleBtn = document.getElementById('soundMonitorToggle');
            this.toggleLabel = document.getElementById('soundMonitorLabel');
            this.status = document.getElementById('soundStatus');
            this.lastAnalyzed = document.getElementById('soundLastAnalyzed');
            this.detectedList = document.getElementById('soundDetected');
            this.serviceNotice = document.getElementById('soundServiceNotice');
            this.alertsSection = document.getElementById('soundAlertsSection');
            this.alertsList = document.getElementById('soundAlerts');
            this.noTargetsNotice = document.getElementById('soundNoTargets');
            this.activeList = document.getElementById('soundActiveDetections');
            this.recentList = document.getElementById('soundRecentAlerts');

            if (!this.toggleBtn) return;   // panel not on this page

            this.monitoring = false;
            this.starting = false;
            this.session = 0;          // bumps on every start/stop; stale responses are ignored
            this.inFlight = false;

            // One state per category: null = IDLE, object = DETECTED.
            this.detections = {};
            CATEGORIES.forEach((c) => { this.detections[c.key] = null; });

            this.nextAlertId = 1;
            this.activeAlerts = [];    // unacknowledged, oldest first
            this.alertCards = new Map();   // alert id -> its card on the panel
            this.recentAlerts = [];    // newest first, one entry per event

            this.alarmTimer = null;    // the one shared alarm; null = silent
            this.alarmGains = new Set();   // beeps currently scheduled, so they can be cut

            // The categories whose events raise alerts.
            this.targets = new Set();
            document.querySelectorAll('input[name="soundTarget"]').forEach((input) => {
                if (input.checked) this.targets.add(input.value);
                input.addEventListener('change', () => this.setTarget(input.value, input.checked));
            });

            if (this.alertsList) {
                this.alertsList.addEventListener('click', (event) => {
                    const button = event.target.closest('button[data-alert-id]');
                    if (button) this.acknowledge(Number(button.dataset.alertId));
                });
            }

            this.renderTargets();
            this.renderAlerts();
            this.renderActiveDetections();
            this.renderRecentAlerts();

            this.toggleBtn.addEventListener('click', () => {
                if (this.monitoring || this.starting) {
                    this.stop('Monitoring off');
                } else {
                    this.start();
                }
            });

            this.setState('idle', 'Monitoring off');
            this.checkHealth();
        }

        get config() {
            return window.GALAMSEY_CONFIG || {};
        }

        get apiUrl() {
            return this.config.SOUND_API_URL || 'http://127.0.0.1:8000';
        }

        get windowSeconds() {
            return this.config.SOUND_WINDOW_SECONDS || DEFAULT_WINDOW_SECONDS;
        }

        // -----------------------------------------------------------
        // Lifecycle
        // -----------------------------------------------------------

        async start() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia ||
                typeof AudioWorkletNode === 'undefined') {
                // Also what an insecure (file://) page looks like.
                this.setState('error', 'Microphone capture is not supported here. ' +
                                       'Open the dashboard over http://127.0.0.1 in Chrome.');
                return;
            }

            const session = ++this.session;
            this.resetDetections();            // every monitoring run starts IDLE
            this.starting = true;
            this.setToggle(true);
            this.setState('analyzing', 'Requesting microphone...');

            try {
                this.stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                });
                if (session !== this.session) {        // switched OFF while the prompt was open
                    this.releaseMicrophone();
                    return;
                }

                this.audioContext = new AudioContext();
                await this.audioContext.audioWorklet.addModule('sound-capture-worklet.js');
                if (session !== this.session) {
                    this.releaseMicrophone();
                    return;
                }

                this.source = this.audioContext.createMediaStreamSource(this.stream);
                this.captureNode = new AudioWorkletNode(this.audioContext, 'sound-capture');
                this.captureNode.port.onmessage = (event) => this.collect(event.data, session);
                this.source.connect(this.captureNode);
                // Connected so the audio graph keeps pulling the node. It
                // outputs silence, so nothing is played back.
                this.captureNode.connect(this.audioContext.destination);
                await this.audioContext.resume();

                this.sampleRate = this.audioContext.sampleRate;
                this.windowSamples = Math.round(this.sampleRate * this.windowSeconds);
                this.window = new Float32Array(this.windowSamples);
                this.windowLength = 0;

                // Microphone unplugged or revoked mid-session.
                this.stream.getAudioTracks().forEach((track) => {
                    track.addEventListener('ended', () => {
                        if (session === this.session) this.stop('Microphone disconnected');
                    });
                });

                this.starting = false;
                this.monitoring = true;
                this.setState('listening', 'Listening...');
                console.log('[galamsey] audio monitoring started (' + this.sampleRate + ' Hz, ' +
                            this.windowSeconds + ' s windows)');
            } catch (error) {
                if (session !== this.session) return;
                console.warn('[galamsey] could not start audio monitoring:', error.name, error.message);
                this.stop(this.microphoneErrorMessage(error), 'error');
            }
        }

        /**
         * Stop capture and release the microphone. The last prediction is
         * left on screen; "Last analyzed" says how old it is. Open alerts
         * stay until acknowledged, but the alarm stops.
         */
        stop(message, state) {
            this.session++;                 // any response still in flight is now ignored
            this.monitoring = false;
            this.starting = false;
            this.inFlight = false;
            this.resetDetections();         // open alerts and Recent Alerts are kept
            // Open alerts stay on screen to acknowledge, but stop sounding.
            this.activeAlerts.forEach((alert) => { alert.sounding = false; });
            this.updateAlarm();
            this.releaseMicrophone();
            this.window = null;
            this.windowLength = 0;
            this.setToggle(false);
            this.setState(state || 'idle', message || 'Monitoring off');
            console.log('[galamsey] audio monitoring stopped');
        }

        releaseMicrophone() {
            if (this.captureNode) {
                this.captureNode.port.onmessage = null;
                this.captureNode.disconnect();
                this.captureNode = null;
            }
            if (this.source) {
                this.source.disconnect();
                this.source = null;
            }
            if (this.stream) {
                // Stopping the tracks is what turns off the browser's
                // "microphone in use" indicator.
                this.stream.getTracks().forEach((track) => track.stop());
                this.stream = null;
            }
            if (this.audioContext) {
                this.audioContext.close().catch(() => {});
                this.audioContext = null;
            }
        }

        microphoneErrorMessage(error) {
            switch (error && error.name) {
                case 'NotAllowedError':
                case 'SecurityError':
                    return 'Microphone permission required. Allow microphone access for this page, then switch monitoring on.';
                case 'NotFoundError':
                case 'OverconstrainedError':
                    return 'Microphone unavailable: no microphone was found.';
                case 'NotReadableError':
                case 'AbortError':
                    return 'Microphone unavailable: it may be in use by another application.';
                default:
                    return 'Could not start audio capture: ' + ((error && error.message) || 'unknown error');
            }
        }

        // -----------------------------------------------------------
        // Windowing
        // -----------------------------------------------------------

        /** Append one batch from the worklet; send each window as it fills. */
        collect(batch, session) {
            if (session !== this.session || !this.window) return;

            let offset = 0;
            while (offset < batch.length) {
                const count = Math.min(batch.length - offset, this.windowSamples - this.windowLength);
                this.window.set(batch.subarray(offset, offset + count), this.windowLength);
                this.windowLength += count;
                offset += count;

                if (this.windowLength === this.windowSamples) {
                    const completed = this.window;
                    this.window = new Float32Array(this.windowSamples);
                    this.windowLength = 0;
                    this.analyzeWindow(completed, session);
                }
            }
        }

        async analyzeWindow(samples, session) {
            if (this.inFlight) {
                console.warn('[galamsey] previous audio window still being analysed; skipping this one');
                return;
            }
            this.inFlight = true;
            this.setState('analyzing', 'Analyzing...');

            // Content-Type is left to the browser so it can add the
            // multipart boundary.
            const form = new FormData();
            form.append('file', encodeWav(samples, this.sampleRate), 'microphone.wav');

            try {
                let response;
                try {
                    response = await fetch(this.apiUrl + '/predict', { method: 'POST', body: form });
                } catch (error) {
                    // Server down, wrong port, or CORS refused. Keep listening:
                    // the next window retries, so monitoring recovers by itself
                    // once the API is back.
                    if (session !== this.session) return;
                    console.warn('[galamsey] sound detection request failed (' + error.message + ')');
                    this.setServiceAvailable(false);
                    this.setState('unavailable', 'Service unavailable. Retrying with the next window...');
                    return;
                }

                const result = await response.json().catch(() => null);
                if (session !== this.session) return;

                if (!response.ok) {
                    this.handleHttpError(response.status, result);
                    return;
                }
                if (!this.isValidResult(result)) {
                    console.warn('[galamsey] unexpected /predict response:', result);
                    this.setState('error', 'Unexpected response from sound detection service');
                    return;
                }

                this.setServiceAvailable(true);
                this.renderResult(result);
                this.lastAnalyzed.textContent = formatTime(new Date());
                this.setState('listening', 'Listening...');
                // Only a validated, successful prediction reaches here.
                this.updateDetections(result);
            } finally {
                if (session === this.session) this.inFlight = false;
            }
        }

        // -----------------------------------------------------------
        // API
        // -----------------------------------------------------------

        /**
         * Optional startup check. Never throws: an unavailable service
         * only shows a notice, the rest of the dashboard is unaffected.
         */
        async checkHealth() {
            try {
                const response = await fetch(this.apiUrl + '/health', { cache: 'no-store' });
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const health = await response.json();
                this.setServiceAvailable(health.model_loaded === true);
            } catch (error) {
                console.warn('[galamsey] sound detection service unavailable (' + error.message + ')');
                this.setServiceAvailable(false);
            }
        }

        handleHttpError(status, body) {
            const detail = body && typeof body.detail === 'string' ? body.detail : 'HTTP ' + status;
            console.warn('[galamsey] sound detection HTTP ' + status + ':', detail);
            if (status === 503) {
                this.setServiceAvailable(false);
                this.setState('unavailable', 'Service unavailable: sound model not loaded');
            } else if (status === 400 || status === 415 || status === 422) {
                // The server's 400 message includes its temp-file path; the
                // full text is in the console above.
                this.setState('error', 'Invalid audio: the service could not process this window');
            } else {
                this.setState('error', 'Sound detection failed (HTTP ' + status + ')');
            }
        }

        isValidResult(result) {
            return !!result &&
                CATEGORIES.every((c) => typeof result[c.key] === 'number' && isFinite(result[c.key])) &&
                Array.isArray(result.detected);
        }

        // -----------------------------------------------------------
        // Detection events
        // -----------------------------------------------------------

        /** Advance each selected category's IDLE/DETECTED machine with one prediction. */
        updateDetections(result) {
            CATEGORIES.forEach((category) => {
                if (!this.targets.has(category.key)) return;

                const probability = result[category.key];
                const state = this.detections[category.key];

                if (probability >= DETECTION_THRESHOLD && !state) {
                    // IDLE -> DETECTED: a new event, and the only place an alert is created.
                    const alert = this.raiseAlert(category, probability);
                    this.detections[category.key] = {
                        startedAt: alert.startedAt,
                        confidence: probability,
                        alert: alert
                    };
                } else if (probability >= DETECTION_THRESHOLD) {
                    // DETECTED -> DETECTED: same event continues, no new alert.
                    state.confidence = probability;
                    state.alert.confidence = probability;
                } else if (state) {
                    // DETECTED -> IDLE: the event ends quietly. Its alert, if
                    // not yet acknowledged, stays until it is.
                    console.log('[galamsey] sound event ended: ' + category.event);
                    this.resetDetection(category.key);
                }
            });
            this.renderAlerts();
            this.renderActiveDetections();
        }

        /** One category back to IDLE without alerting. */
        resetDetection(key) {
            const state = this.detections[key];
            if (state) state.alert.ongoing = false;    // the alert no longer tracks a live event
            this.detections[key] = null;
        }

        /** Every category back to IDLE. Alerts and Recent Alerts are untouched. */
        resetDetections() {
            CATEGORIES.forEach((c) => this.resetDetection(c.key));
            this.renderAlerts();
            this.renderActiveDetections();
        }

        raiseAlert(category, probability) {
            const alert = {
                id: this.nextAlertId++,
                key: category.key,
                message: category.event,
                startConfidence: probability,   // what Recent Alerts shows
                confidence: probability,        // follows the event while it lasts
                startedAt: new Date(),
                ongoing: true,
                sounding: true,
                acknowledgedAt: null
            };
            console.log('[galamsey] sound alert: ' + alert.message + ' (' + formatPercent(probability) + ')');

            this.activeAlerts.push(alert);
            this.recentAlerts.unshift(alert);
            if (this.recentAlerts.length > MAX_RECENT_ALERTS) this.recentAlerts.length = MAX_RECENT_ALERTS;
            this.renderRecentAlerts();
            this.updateAlarm();
            return alert;
        }

        /**
         * Close one alert. Silences it and moves it to history only: the
         * category's detection state is deliberately left alone, because
         * the sound may well still be there.
         */
        acknowledge(id) {
            const index = this.activeAlerts.findIndex((a) => a.id === id);
            if (index === -1) return;
            const alert = this.activeAlerts[index];
            alert.acknowledgedAt = new Date();
            alert.sounding = false;
            this.activeAlerts.splice(index, 1);
            console.log('[galamsey] sound alert acknowledged: ' + alert.message);

            this.updateAlarm();
            this.renderAlerts();
            this.renderRecentAlerts();
        }

        // -----------------------------------------------------------
        // Alarm
        // -----------------------------------------------------------

        /**
         * Start or stop the one shared alarm. It runs while monitoring is
         * on and at least one open alert is still sounding, so two open
         * alerts never mean two overlapping loops.
         */
        updateAlarm() {
            const wanted = this.monitoring && this.activeAlerts.some((a) => a.sounding);
            if (wanted && !this.alarmTimer) {
                this.alarmTimer = setInterval(() => this.playBeep(), ALARM_PERIOD_MS);
                this.playBeep();
            } else if (!wanted && this.alarmTimer) {
                clearInterval(this.alarmTimer);
                this.alarmTimer = null;
                // Cut a beep that is already scheduled rather than let it finish.
                this.alarmGains.forEach((gain) => gain.disconnect());
                this.alarmGains.clear();
            }
        }

        /**
         * One beep-beep through the monitoring AudioContext. That context
         * was created by the ON click, so autoplay rules are already
         * satisfied, and it is closed on OFF. (script.js's playAlertSound()
         * opens a new AudioContext per call and never closes it, which
         * would accumulate over a long alarm.) The Settings > Sound Alerts
         * switch is checked on every beep, so flipping it takes effect
         * straight away. A failure here is logged and otherwise ignored:
         * the on-screen alert and the monitoring loop do not depend on it.
         */
        playBeep() {
            const dashboard = window.detectorInstance;
            if (dashboard && dashboard.settings && dashboard.settings.soundAlerts === false) return;

            const ctx = this.audioContext;
            if (!ctx || ctx.state !== 'running') return;

            let gain = null;
            try {
                gain = ctx.createGain();
                gain.gain.value = 0;
                gain.connect(ctx.destination);
                this.alarmGains.add(gain);
                const start = ctx.currentTime + 0.02;
                BEEP_OFFSETS.forEach((offset, i) => {
                    const osc = ctx.createOscillator();
                    osc.type = 'sine';
                    osc.frequency.value = 880;
                    osc.connect(gain);
                    gain.gain.setValueAtTime(0.25, start + offset);
                    gain.gain.exponentialRampToValueAtTime(0.001, start + offset + BEEP_SECONDS);
                    osc.start(start + offset);
                    osc.stop(start + offset + BEEP_SECONDS);
                    if (i === BEEP_OFFSETS.length - 1) {
                        osc.onended = () => {
                            gain.disconnect();
                            this.alarmGains.delete(gain);
                        };
                    }
                });
            } catch (error) {
                console.warn('[galamsey] could not play alert tone:', error);
                if (gain) {
                    gain.disconnect();
                    this.alarmGains.delete(gain);
                }
            }
        }

        // -----------------------------------------------------------
        // Rendering
        // -----------------------------------------------------------

        /**
         * Open alerts, newest first. Cards are created once and updated in
         * place, so a confidence update never replaces an Acknowledge
         * button the user is about to press.
         */
        renderAlerts() {
            if (!this.alertsList) return;
            const open = new Set(this.activeAlerts.map((a) => a.id));
            this.alertCards.forEach((card, id) => {
                if (!open.has(id)) {
                    card.remove();
                    this.alertCards.delete(id);
                }
            });

            this.activeAlerts.forEach((alert) => {
                let card = this.alertCards.get(alert.id);
                if (!card) {
                    card = createAlertCard(alert);
                    this.alertCards.set(alert.id, card);
                    this.alertsList.prepend(card);
                }
                card.querySelector('.sound-alert-card__confidence').textContent =
                    'Confidence: ' + formatPercent(alert.confidence);
                card.querySelector('.sound-alert-card__ongoing').hidden = !alert.ongoing;
            });

            this.alertsSection.hidden = this.activeAlerts.length === 0;
        }

        /** Selected categories that are currently DETECTED. */
        renderActiveDetections() {
            if (!this.activeList) return;
            this.activeList.innerHTML = '';
            const active = CATEGORIES.filter((c) => this.detections[c.key]);
            if (active.length === 0) {
                this.activeList.appendChild(createRow(['None']));
                return;
            }
            active.forEach((c) => {
                const state = this.detections[c.key];
                const row = createRow([c.display, formatPercent(state.confidence),
                                       'Since ' + formatTime(state.startedAt)]);
                row.classList.add('sound-active');
                this.activeList.appendChild(row);
            });
        }

        renderRecentAlerts() {
            if (!this.recentList) return;
            this.recentList.innerHTML = '';
            if (this.recentAlerts.length === 0) {
                this.recentList.appendChild(createRow(['No alerts yet']));
                return;
            }
            this.recentAlerts.forEach((alert) => {
                const item = createRow([formatTime(alert.startedAt), alert.message,
                                        formatPercent(alert.startConfidence)]);
                item.classList.add('sound-recent');
                item.firstChild.className = 'sound-recent__time';

                const status = document.createElement('i');
                const acknowledged = alert.acknowledgedAt !== null;
                status.className = 'sound-recent__status fas ' +
                    (acknowledged ? 'fa-check sound-recent__status--ack' : 'fa-bell');
                status.title = acknowledged
                    ? 'Acknowledged at ' + formatTime(alert.acknowledgedAt)
                    : 'Not yet acknowledged';
                status.setAttribute('aria-label', status.title);
                item.appendChild(status);
                this.recentList.appendChild(item);
            });
        }

        renderResult(result) {
            CATEGORIES.forEach((c) => {
                const el = document.getElementById('soundProb-' + c.key);
                // Raw independent probability, as returned. Not normalised.
                el.textContent = formatPercent(result[c.key]);
                el.classList.toggle('sound-prob--detected', result.detected.includes(c.key));
            });

            this.detectedList.innerHTML = '';
            const names = result.detected.map((key) => {
                const category = CATEGORIES.find((c) => c.key === key);
                return category ? category.display : key;
            });
            (names.length ? names : ['None']).forEach((name) => {
                this.detectedList.appendChild(createRow([name]));
            });
        }

        /**
         * Tick or untick one target. Either way the category starts again
         * from IDLE, so a stale event can neither carry over nor alert. An
         * open alert for an unticked category stays until acknowledged: it
         * records an event that did happen.
         */
        setTarget(key, selected) {
            this.resetDetection(key);
            if (selected) this.targets.add(key);
            else this.targets.delete(key);
            this.renderTargets();
            this.renderAlerts();
            this.renderActiveDetections();
        }

        renderTargets() {
            CATEGORIES.forEach((c) => {
                const row = document.getElementById('soundRow-' + c.key);
                if (row) row.classList.toggle('sound-row--target', this.targets.has(c.key));
            });
            if (this.noTargetsNotice) this.noTargetsNotice.hidden = this.targets.size > 0;
        }

        setToggle(on) {
            this.toggleBtn.classList.toggle('sound-toggle--on', on);
            this.toggleBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
            this.toggleLabel.textContent = on ? 'ON' : 'OFF';
        }

        setState(state, message) {
            this.status.textContent = message;
            this.status.dataset.state = state;
        }

        setServiceAvailable(available) {
            if (this.serviceNotice) this.serviceNotice.hidden = available;
        }
    }

    function formatPercent(probability) {
        return (probability * 100).toFixed(1) + '%';
    }

    function formatTime(date) {
        return date.toLocaleTimeString([], { hour12: false });
    }

    /** A .tool-item row with one <span> per text. */
    function createRow(texts) {
        const item = document.createElement('div');
        item.className = 'tool-item';
        texts.forEach((text) => {
            const span = document.createElement('span');
            span.textContent = text;
            item.appendChild(span);
        });
        return item;
    }

    /** The persistent card for one open alert. Text only, no innerHTML. */
    function createAlertCard(alert) {
        const card = document.createElement('div');
        card.className = 'sound-alert-card';

        const title = document.createElement('div');
        title.className = 'sound-alert-card__title';
        const icon = document.createElement('i');
        icon.className = 'fas fa-triangle-exclamation';
        const ongoing = document.createElement('span');
        ongoing.className = 'sound-alert-card__ongoing';
        ongoing.textContent = 'Ongoing';
        title.append(icon, ' DETECTION ALERT ', ongoing);

        const lines = [
            ['sound-alert-card__message', alert.message],
            ['sound-alert-card__confidence', ''],
            ['sound-alert-card__started', 'Started: ' + formatTime(alert.startedAt)]
        ].map(([className, text]) => {
            const line = document.createElement('div');
            line.className = className;
            line.textContent = text;
            return line;
        });

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'action-btn sound-alert-card__ack';
        button.dataset.alertId = String(alert.id);
        button.textContent = 'Acknowledge';

        card.append(title, ...lines, button);
        return card;
    }

    /**
     * Wrap float samples in [-1, 1] as a 16-bit PCM mono WAV file.
     * 44-byte RIFF header followed by little-endian samples.
     */
    function encodeWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeString = (offset, text) => {
            for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);              // fmt chunk size
        view.setUint16(20, 1, true);               // PCM
        view.setUint16(22, 1, true);               // mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);  // byte rate
        view.setUint16(32, 2, true);               // block align
        view.setUint16(34, 16, true);              // bits per sample
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    document.addEventListener('DOMContentLoaded', () => {
        window.soundDetectionInstance = new SoundDetection();
    });
})();
