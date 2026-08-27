/**
 * TEMPLATE — copy this file to config.local.js and fill in your key.
 *
 *     cp config.local.example.js config.local.js
 *
 * config.local.js is git-ignored. This template is committed so that
 * anyone cloning the project knows exactly which values they must set.
 *
 * A browser Google Maps key CANNOT be kept secret — it is delivered to
 * every visitor's browser. Protect it in Google Cloud Console:
 *   1. APIs & Services > Credentials > your key
 *   2. Application restrictions -> HTTP referrers
 *        http://localhost:5501/*
 *        http://127.0.0.1:5501/*
 *   3. API restrictions -> Maps JavaScript API, Geocoding API, Places API
 *   4. Billing > Budgets & alerts -> set a low daily cap
 */
window.GALAMSEY_LOCAL_CONFIG = {
    GOOGLE_MAPS_API_KEY: 'PUT-YOUR-KEY-HERE'
};
