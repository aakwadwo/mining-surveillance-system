/**
 * Galamsey AI Detector — frontend configuration
 * =============================================
 *
 * THIS FILE IS COMMITTED AND MUST NEVER CONTAIN A REAL API KEY.
 *
 * Machine-specific values (your Google Maps key) belong in
 * config.local.js, which is listed in .gitignore. That file is loaded
 * after this one and its values win. If it is absent the app still
 * runs; only the map is disabled.
 *
 * --------------------------------------------------------------
 * WHY THERE IS NO BUILD SYSTEM HERE
 * --------------------------------------------------------------
 * This frontend is plain HTML + CSS + JavaScript with no npm, no
 * bundler and no framework. That is a deliberate choice, not an
 * oversight.
 *
 * A build system (Vite, webpack, Next.js) earns its place when you need
 * module bundling, JSX/TypeScript compilation, tree shaking, or
 * dependency management for dozens of packages. This app has three
 * script files and two CDN links. Adding a bundler would introduce a
 * node_modules folder, a lock file, a compile step between editing and
 * seeing the result, and a whole new category of things that can break
 * — in exchange for nothing.
 *
 * A build step also would NOT hide the Maps key. Anything the browser
 * can read, the user can read. Bundling only makes secrets harder to
 * find, never actually secret.
 *
 * --------------------------------------------------------------
 * WHY THE BACKEND IS ON A DIFFERENT PORT
 * --------------------------------------------------------------
 * The frontend is static files served by a plain file server on 5501.
 * The backend is a Python program on 5004. They are two separate
 * programs, and a port is just the numbered door on this machine that
 * a given program is listening at. Different program, different door.
 */

window.GALAMSEY_CONFIG = {
    // -----------------------------------------------------------
    // Backend API
    // -----------------------------------------------------------
    // Where backend/app.py is listening. Must match config.py PORT.
    BACKEND_URL: 'http://127.0.0.1:5004',

    // Poll interval for pulling detections from the backend (ms).
    POLL_INTERVAL_MS: 5000,

    // Send detection records to the backend as they occur.
    SEND_DETECTIONS: true,

    // -----------------------------------------------------------
    // Google Maps
    // -----------------------------------------------------------
    // Leave EMPTY here. Put the real key in config.local.js.
    GOOGLE_MAPS_API_KEY: '',
    GOOGLE_MAPS_LIBRARIES: 'drawing,geometry,places',

    // Default map centre: Accra, Ghana.
    DEFAULT_CENTER: { lat: 5.6037, lng: -0.1870 },
    DEFAULT_ZOOM: 15,

    // -----------------------------------------------------------
    // Detection classes — MUST match backend/taxonomy.py
    // -----------------------------------------------------------
    // The backend sends class ids. These names are for display only.
    // Do not invent classes here: the model emits exactly these six.
    CLASSES: [
        { id: 0, name: 'ExcavationMachinery', display: 'Excavation Machinery' },
        { id: 1, name: 'MiningTool',          display: 'Mining Tool' },
        { id: 2, name: 'Person',              display: 'Person' },
        { id: 3, name: 'ProcessingEquiment',  display: 'Processing Equipment' },
        { id: 4, name: 'Structure',           display: 'Structure' },
        { id: 5, name: 'Vehicle',             display: 'Vehicle' }
    ],

    // -----------------------------------------------------------
    // Legacy drone settings (NOT currently used)
    // -----------------------------------------------------------
    // The original frontend targeted a drone at 192.168.1.100:8080 that
    // does not exist. Kept only so the Settings dialog still has values
    // to show. Real detection traffic goes to BACKEND_URL above.
    DRONE_IP: '192.168.1.100',
    DRONE_PORT: 8080
};

/**
 * Merge config.local.js over the defaults, then expose helpers.
 * Called by index.html and map-popup.html after both files load.
 */
window.GALAMSEY_RESOLVE_CONFIG = function () {
    const cfg = window.GALAMSEY_CONFIG;
    const local = window.GALAMSEY_LOCAL_CONFIG || {};
    Object.assign(cfg, local);
    return cfg;
};

/**
 * Load the Google Maps JavaScript API.
 *
 * The inherited index.html loaded Maps with:
 *     <script async defer src="...&callback=initMap">
 * while window.initMap was only assigned later, inside a
 * DOMContentLoaded handler. Whether that worked depended on which
 * finished first — a race. On a cold cache Maps often called initMap
 * before it existed, and the map silently never appeared.
 *
 * Here the callback is guaranteed to exist before the script tag is
 * created, so the race cannot happen.
 *
 * @param {string} callbackName global function Maps should call
 * @returns {boolean} false if no API key is configured
 */
window.GALAMSEY_LOAD_MAPS = function (callbackName) {
    const cfg = window.GALAMSEY_RESOLVE_CONFIG();

    if (!cfg.GOOGLE_MAPS_API_KEY) {
        console.warn(
            '[galamsey] No Google Maps API key configured. ' +
            'Copy config.local.example.js to config.local.js and add your key. ' +
            'The rest of the dashboard works without it.'
        );
        return false;
    }

    if (typeof window[callbackName] !== 'function') {
        console.error('[galamsey] Maps callback "' + callbackName + '" is not defined yet.');
        return false;
    }

    const script = document.createElement('script');
    script.src =
        'https://maps.googleapis.com/maps/api/js' +
        '?key=' + encodeURIComponent(cfg.GOOGLE_MAPS_API_KEY) +
        '&callback=' + encodeURIComponent(callbackName) +
        '&libraries=' + encodeURIComponent(cfg.GOOGLE_MAPS_LIBRARIES) +
        '&loading=async';
    script.async = true;
    script.onerror = function () {
        console.error('[galamsey] Google Maps failed to load. Check the key and its restrictions.');
    };
    document.head.appendChild(script);
    return true;
};
