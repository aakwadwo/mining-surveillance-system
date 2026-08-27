/**
 * OPTIONAL FALLBACK — you normally do NOT need this file.
 *
 * The Google Maps key lives in the git-ignored .env at the PROJECT ROOT
 * (see .env.example). The backend reads it from the environment and
 * serves it to the browser at GET /client-config, so no credential sits
 * in any frontend file.
 *
 * Use this file ONLY if you want the map to work while running the
 * frontend WITHOUT the backend:
 *
 *     cp config.local.example.js config.local.js
 *
 * config.local.js is git-ignored. Values here are overridden by the
 * backend's /client-config when it is reachable.
 *
 * Either way, a browser Google Maps key CANNOT be kept secret — it is
 * delivered to every visitor. Protect it in Google Cloud Console:
 *   1. Credentials > key > Application restrictions > HTTP referrers
 *        http://localhost:5501/*
 *   2. API restrictions > Maps JavaScript API, Geocoding API, Places API
 *   3. Billing > Budgets & alerts > low daily cap
 */
window.GALAMSEY_LOCAL_CONFIG = {
    GOOGLE_MAPS_API_KEY: 'PUT-YOUR-KEY-HERE'
};
