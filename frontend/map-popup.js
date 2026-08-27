// Map Popup JavaScript - Handles the popup window map functionality
let popupMap = null;
let popupDroneMarker = null;
let popupDronePath = null;
let popupDronePathCoordinates = [];
let popupGeofencePolygons = [];
let popupDrawingManager = null;
let popupIsGeofenceMode = false;
let popupCurrentDronePosition = { lat: 5.6037, lng: -0.1870, altitude: 0, heading: 0 };
let popupGeofences = [];
let popupGeocoder = null;
let popupSearchMarker = null;
let popupUseDeviceLocation = true;
let popupGeolocationWatchId = null;
let popupFirstLocationReceived = false;
let popupChannel = null;
let popupIsInsideGeofence = false;
let popupIsInsideRestrictedArea = false;

// Load initial state from sessionStorage
let mapState = null;
try {
    const stateStr = sessionStorage.getItem('popupMapState');
    if (stateStr) {
        mapState = JSON.parse(stateStr);
        popupCurrentDronePosition = mapState.dronePosition || popupCurrentDronePosition;
    }
} catch (e) {
    console.error('Failed to load map state:', e);
}

// Load geofences from localStorage
try {
    const saved = localStorage.getItem('geofences');
    if (saved) {
        popupGeofences = JSON.parse(saved);
    }
} catch (e) {
    console.error('Failed to load geofences:', e);
}

// Make initPopupMap globally accessible for Google Maps callback
window.initPopupMap = function() {
    console.log('initPopupMap called');
    
    // Wait a bit to ensure DOM is ready
    setTimeout(() => {
        if (typeof google === 'undefined' || !google.maps) {
            console.error('Google Maps API not loaded');
            const mapView = document.getElementById('popupMapView');
            if (mapView) {
                mapView.innerHTML = 
                    '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #FF4444; padding: 2rem; text-align: center;">Google Maps API not loaded. Please check your API key.</div>';
            }
            return;
        }

        const mapView = document.getElementById('popupMapView');
        if (!mapView) {
            console.error('Map view container not found');
            return;
        }
        
        // Ensure map container has dimensions
        if (mapView.offsetWidth === 0 || mapView.offsetHeight === 0) {
            console.warn('Map container has no dimensions, setting explicit size');
            mapView.style.width = '100%';
            mapView.style.height = '100%';
            mapView.style.minHeight = '400px';
        }

        // Get map state
        if (!mapState) {
            mapState = {
                lat: 5.6037,
                lng: -0.1870,
                zoom: 15,
                mapType: 'SATELLITE'
            };
        }

        // Create map
        const mapTypeMap = {
            'SATELLITE': google.maps.MapTypeId.SATELLITE,
            'TERRAIN': google.maps.MapTypeId.TERRAIN,
            'HYBRID': google.maps.MapTypeId.HYBRID,
            'ROADMAP': google.maps.MapTypeId.ROADMAP
        };
        const mapTypeId = mapTypeMap[mapState.mapType] || google.maps.MapTypeId.SATELLITE;
        
        try {
            // If using device location, don't center on default location yet
            // It will center when first location is received
            const initialCenter = popupUseDeviceLocation ? 
                { lat: mapState.lat, lng: mapState.lng } : 
                { lat: mapState.lat, lng: mapState.lng };
            
            popupMap = new google.maps.Map(mapView, {
                center: initialCenter,
                zoom: mapState.zoom,
                mapTypeId: mapTypeId,
                disableDefaultUI: false,
                zoomControl: true,
                mapTypeControl: true,
                fullscreenControl: true,
                styles: [
                    {
                        featureType: 'all',
                        elementType: 'labels',
                        stylers: [{ visibility: 'on' }]
                    }
                ]
            });
            console.log('Popup map initialized successfully');
        } catch (error) {
            console.error('Error initializing map:', error);
            mapView.innerHTML = 
                '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #FF4444; padding: 2rem; text-align: center;">Error initializing map: ' + error.message + '</div>';
            return;
        }

        // Create drone marker
        createPopupDroneMarker();
        
        // Initialize drone path
        popupDronePath = new google.maps.Polyline({
            path: popupDronePathCoordinates,
            geodesic: true,
            strokeColor: '#00FF99',
            strokeOpacity: 0.8,
            strokeWeight: 3,
            map: popupMap
        });

        // Render geofences
        renderPopupGeofences();

        // Setup drawing manager
        setupPopupDrawingManager();

        // Initialize Geocoding API
        popupGeocoder = new google.maps.Geocoder();

        // Setup controls
        setupPopupControls();

        // Set up communication with parent tab using BroadcastChannel
        popupChannel = new BroadcastChannel('mapTabChannel');
        
        // Listen for position updates from parent tab
        popupChannel.onmessage = (event) => {
            if (event.data.type === 'dronePosition') {
                // Stop device location if receiving data from parent
                if (popupUseDeviceLocation && popupGeolocationWatchId !== null) {
                    stopPopupDeviceLocationTracking();
                    popupUseDeviceLocation = false;
                }
                updatePopupDronePosition(event.data.position);
            }
        };
        
        document.getElementById('popupCloseBtn').addEventListener('click', () => {
            // Notify parent tab that this tab is closing
            if (popupChannel) {
                popupChannel.postMessage({ type: 'tabClosing' });
                popupChannel.close();
            }
            stopPopupDeviceLocationTracking();
            window.close();
        });
        
        // Also handle window beforeunload to notify parent when tab is closed
        window.addEventListener('beforeunload', () => {
            if (popupChannel) {
                popupChannel.postMessage({ type: 'tabClosing' });
                popupChannel.close();
            }
            stopPopupDeviceLocationTracking();
        });

        // Request initial position from parent tab
        if (popupChannel) {
            popupChannel.postMessage({ type: 'requestPosition' });
        }

        // Start device location tracking if enabled
        if (popupUseDeviceLocation) {
            setTimeout(() => {
                startPopupDeviceLocationTracking();
            }, 500);
        }
    }, 100);
};

// Fallback: If Google Maps API loads before this script, try to initialize
if (typeof google !== 'undefined' && google.maps && typeof window.initPopupMap === 'function') {
    window.initPopupMap();
}

function createPopupDroneIcon(color) {
    const canvas = document.createElement('canvas');
    const size = 64;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const centerX = size / 2;
    const centerY = size / 2;
    const bodySize = 14;
    const armLength = 20;
    const propSize = 9;
    const armWidth = 4;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = armWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Top arm
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - bodySize/2);
    ctx.lineTo(centerX, centerY - armLength);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(centerX, centerY - armLength, propSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - armLength - propSize);
    ctx.lineTo(centerX, centerY - armLength + propSize);
    ctx.moveTo(centerX - propSize, centerY - armLength);
    ctx.lineTo(centerX + propSize, centerY - armLength);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY - armLength, propSize, 0, Math.PI * 2);
    ctx.stroke();

    // Right arm
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = armWidth;
    ctx.beginPath();
    ctx.moveTo(centerX + bodySize/2, centerY);
    ctx.lineTo(centerX + armLength, centerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX + armLength, centerY, propSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX + armLength, centerY - propSize);
    ctx.lineTo(centerX + armLength, centerY + propSize);
    ctx.moveTo(centerX + armLength - propSize, centerY);
    ctx.lineTo(centerX + armLength + propSize, centerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX + armLength, centerY, propSize, 0, Math.PI * 2);
    ctx.stroke();

    // Bottom arm
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = armWidth;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY + bodySize/2);
    ctx.lineTo(centerX, centerY + armLength);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY + armLength, propSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY + armLength - propSize);
    ctx.lineTo(centerX, centerY + armLength + propSize);
    ctx.moveTo(centerX - propSize, centerY + armLength);
    ctx.lineTo(centerX + propSize, centerY + armLength);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY + armLength, propSize, 0, Math.PI * 2);
    ctx.stroke();

    // Left arm
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = armWidth;
    ctx.beginPath();
    ctx.moveTo(centerX - bodySize/2, centerY);
    ctx.lineTo(centerX - armLength, centerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX - armLength, centerY, propSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - armLength, centerY - propSize);
    ctx.lineTo(centerX - armLength, centerY + propSize);
    ctx.moveTo(centerX - armLength - propSize, centerY);
    ctx.lineTo(centerX - armLength + propSize, centerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX - armLength, centerY, propSize, 0, Math.PI * 2);
    ctx.stroke();

    // Central body
    ctx.fillStyle = color;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(centerX, centerY, bodySize/2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(centerX - 3, centerY - 3, 1.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(centerX + 3, centerY - 3, 1.5, 0, Math.PI * 2);
    ctx.fill();

    return {
        url: canvas.toDataURL(),
        scaledSize: new google.maps.Size(size, size),
        anchor: new google.maps.Point(size/2, size/2),
        rotation: popupCurrentDronePosition.heading
    };
}

function createPopupDroneMarker() {
    if (!popupMap) return;
    
    const droneIcon = createPopupDroneIcon('#00FF99');
    
    popupDroneMarker = new google.maps.Marker({
        position: popupCurrentDronePosition,
        map: popupMap,
        icon: droneIcon,
        title: 'Drone Position',
        zIndex: 1000
    });
}

function updatePopupDronePosition(position) {
    if (!popupMap) {
        console.warn('Map not initialized yet');
        return;
    }

    if (!popupDroneMarker) {
        createPopupDroneMarker();
        if (!popupDroneMarker) {
            console.error('Failed to create drone marker');
            return;
        }
    }

    // Validate position data
    if (typeof position.lat !== 'number' || typeof position.lng !== 'number') {
        console.error('Invalid position data:', position);
        return;
    }

    // If this is the first position update, center the map on it
    if (!popupFirstLocationReceived) {
        popupFirstLocationReceived = true;
        popupMap.setCenter({ lat: position.lat, lng: position.lng });
        popupMap.setZoom(17);
        console.log('Popup map centered on first drone position:', position.lat, position.lng);
    }

    popupCurrentDronePosition = {
        lat: position.lat,
        lng: position.lng,
        altitude: position.altitude !== undefined ? position.altitude : popupCurrentDronePosition.altitude,
        heading: position.heading !== undefined ? position.heading : popupCurrentDronePosition.heading
    };
    
    popupDroneMarker.setPosition(popupCurrentDronePosition);
    
    // Update icon with current heading
    let color = '#00FF99';
    if (popupIsInsideRestrictedArea) {
        color = '#FF4444';
    } else if (!popupIsInsideGeofence) {
        color = '#FFA500';
    }
    
    const newIcon = createPopupDroneIcon(color);
    newIcon.rotation = popupCurrentDronePosition.heading;
    popupDroneMarker.setIcon(newIcon);
    
    popupDronePathCoordinates.push({ lat: popupCurrentDronePosition.lat, lng: popupCurrentDronePosition.lng });
    if (popupDronePathCoordinates.length > 1000) {
        popupDronePathCoordinates.shift();
    }
    if (popupDronePath) {
        popupDronePath.setPath(popupDronePathCoordinates);
    }

    // Check geofence violations
    checkPopupGeofenceViolations();
    
    // Update marker color
    updatePopupDroneMarkerColor();
}

function checkPopupGeofenceViolations() {
    if (!popupMap || !popupDroneMarker || popupGeofencePolygons.length === 0) {
        popupIsInsideGeofence = true;
        return;
    }

    const dronePosition = new google.maps.LatLng(
        popupCurrentDronePosition.lat,
        popupCurrentDronePosition.lng
    );

    let insideAllowedGeofence = false;
    let insideRestrictedGeofence = false;
    let currentGeofence = null;

    popupGeofencePolygons.forEach((polygon) => {
        if (google.maps.geometry.poly.containsLocation(dronePosition, polygon)) {
            const geofenceId = polygon.__geofenceId;
            const geofence = popupGeofences.find(g => g.id === geofenceId);
            
            if (geofence) {
                currentGeofence = geofence;
                if (geofence.type === 'restricted') {
                    insideRestrictedGeofence = true;
                } else {
                    insideAllowedGeofence = true;
                }
            }
        }
    });

    // Handle restricted area violation
    if (insideRestrictedGeofence) {
        if (!popupIsInsideRestrictedArea) {
            const geofenceName = currentGeofence ? currentGeofence.name : 'restricted area';
            alert(`🚨 CRITICAL: DRONE ENTERED RESTRICTED MINING AREA "${geofenceName}"!`);
        }
        popupIsInsideRestrictedArea = true;
        popupIsInsideGeofence = false;
    } else if (popupIsInsideRestrictedArea) {
        alert('✅ Drone exited restricted mining area');
        popupIsInsideRestrictedArea = false;
    }

    // Handle allowed area
    if (insideAllowedGeofence && !insideRestrictedGeofence) {
        popupIsInsideGeofence = true;
    } else if (!insideRestrictedGeofence && !insideAllowedGeofence) {
        popupIsInsideGeofence = false;
    }
}

function updatePopupDroneMarkerColor() {
    if (!popupDroneMarker) return;

    let color = '#FFA500'; // Default: orange for outside all geofences
    if (popupIsInsideRestrictedArea) {
        color = '#FF4444'; // Red for restricted area
    } else if (popupIsInsideGeofence) {
        color = '#00FF99'; // Green for allowed area
    }

    const newIcon = createPopupDroneIcon(color);
    newIcon.rotation = popupCurrentDronePosition.heading;
    popupDroneMarker.setIcon(newIcon);
}

function getPopupGeofenceColors(type) {
    if (type === 'restricted') {
        return {
            fill: '#FF4444',
            stroke: '#FF4444'
        };
    } else {
        return {
            fill: '#00FF99',
            stroke: '#00FF99'
        };
    }
}

function renderPopupGeofences() {
    if (!popupMap || typeof google === 'undefined') return;
    popupGeofencePolygons.forEach(p => p.setMap(null));
    popupGeofencePolygons = [];
    popupGeofences.forEach(geofence => {
        if (!geofence.coordinates || geofence.coordinates.length < 3) return;
        
        const geofenceType = geofence.type || 'allowed';
        const colors = getPopupGeofenceColors(geofenceType);
        
        const polygon = new google.maps.Polygon({
            paths: geofence.coordinates.map(c => new google.maps.LatLng(c.lat, c.lng)),
            fillColor: colors.fill,
            fillOpacity: 0.15,
            strokeColor: colors.stroke,
            strokeWeight: 2,
            editable: true,
            map: popupMap
        });
        polygon.__geofenceId = geofence.id;
        geofence.polygon = polygon;
        attachPolygonEvents(polygon, geofence);
        popupGeofencePolygons.push(polygon);
    });
}

function setupPopupDrawingManager() {
    if (!popupMap || typeof google === 'undefined') return;
    popupDrawingManager = new google.maps.drawing.DrawingManager({
        drawingMode: null,
        drawingControl: false,
        polygonOptions: {
            fillColor: '#00FF99',
            fillOpacity: 0.2,
            strokeColor: '#00FF99',
            strokeWeight: 2,
            clickable: false,
            editable: true
        }
    });
    popupDrawingManager.setMap(popupMap);
    google.maps.event.addListener(popupDrawingManager, 'polygoncomplete', (polygon) => {
        const name = prompt('Name this geofence:') || `Geofence ${new Date().toLocaleString()}`;
        
        // Ask for geofence type
        const typeChoice = prompt('Select geofence type:\n1. Allowed (Green) - Mining permitted\n2. Restricted (Red) - Mining prohibited\n\nEnter 1 or 2:');
        let geofenceType = 'allowed';
        if (typeChoice === '2' || typeChoice?.toLowerCase() === 'restricted' || typeChoice?.toLowerCase() === 'red') {
            geofenceType = 'restricted';
        }
        
        const path = polygon.getPath();
        const coordinates = [];
        path.forEach((latLng) => {
            coordinates.push({ lat: latLng.lat(), lng: latLng.lng() });
        });
        const geofence = { id: Date.now(), name, type: geofenceType, coordinates, polygon: polygon };
        popupGeofences.push(geofence);
        
        // Style the polygon based on type
        const colors = getPopupGeofenceColors(geofenceType);
        polygon.setOptions({
            fillColor: colors.fill,
            fillOpacity: 0.15,
            strokeColor: colors.stroke,
            strokeWeight: 2,
            editable: true,
            draggable: false
        });
        
        polygon.__geofenceId = geofence.id;
        attachPolygonEvents(polygon, geofence);
        popupGeofencePolygons.push(polygon);
        
        // Add edit listeners
        path.addListener('set_at', () => {
            const coords = [];
            path.forEach((latLng) => {
                coords.push({ lat: latLng.lat(), lng: latLng.lng() });
            });
            geofence.coordinates = coords;
            try {
                const geofencesToSave = popupGeofences.map(g => ({
                    id: g.id,
                    name: g.name,
                    type: g.type || 'allowed',
                    coordinates: g.coordinates
                }));
                localStorage.setItem('geofences', JSON.stringify(geofencesToSave));
            } catch (e) {
                console.error('Failed to save geofence:', e);
            }
        });
        path.addListener('insert_at', () => {
            const coords = [];
            path.forEach((latLng) => {
                coords.push({ lat: latLng.lat(), lng: latLng.lng() });
            });
            geofence.coordinates = coords;
            try {
                const geofencesToSave = popupGeofences.map(g => ({
                    id: g.id,
                    name: g.name,
                    type: g.type || 'allowed',
                    coordinates: g.coordinates
                }));
                localStorage.setItem('geofences', JSON.stringify(geofencesToSave));
            } catch (e) {
                console.error('Failed to save geofence:', e);
            }
        });
        
        try {
            const geofencesToSave = popupGeofences.map(g => ({
                id: g.id,
                name: g.name,
                type: g.type || 'allowed',
                coordinates: g.coordinates
            }));
            localStorage.setItem('geofences', JSON.stringify(geofencesToSave));
        } catch (e) {
            console.error('Failed to save geofence:', e);
        }
        
        renderPopupGeofences();
        popupDrawingManager.setDrawingMode(null);
        popupIsGeofenceMode = false;
        
        const saveBtn = document.getElementById('popupGeofenceSaveBtn');
        const cancelBtn = document.getElementById('popupGeofenceCancelBtn');
        if (saveBtn) saveBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';
    });
}

function attachPolygonEvents(polygon, geofence) {
    polygon.addListener('click', (event) => {
        const confirmed = confirm(`Delete geofence "${geofence.name}"?`);
        if (confirmed) {
            removeGeofenceById(geofence.id);
        }
    });
}

function removeGeofenceById(geofenceId) {
    popupGeofences = popupGeofences.filter(g => g.id !== geofenceId);
    try {
        const geofencesToSave = popupGeofences.map(g => ({
            id: g.id,
            name: g.name,
            type: g.type || 'allowed',
            coordinates: g.coordinates
        }));
        localStorage.setItem('geofences', JSON.stringify(geofencesToSave));
    } catch (e) {
        console.error('Failed to save geofence:', e);
    }

    // Remove polygon from map
    popupGeofencePolygons.forEach((polygon, index) => {
        if (polygon.__geofenceId === geofenceId) {
            polygon.setMap(null);
            popupGeofencePolygons.splice(index, 1);
        }
    });

    renderPopupGeofences();
    alert('Geofence deleted successfully');
}

// Setup popup controls
function setupPopupControls() {
    // Zoom controls
    document.getElementById('popupZoomIn').addEventListener('click', () => {
        if (popupMap) popupMap.setZoom(popupMap.getZoom() + 1);
    });
    
    document.getElementById('popupZoomOut').addEventListener('click', () => {
        if (popupMap) popupMap.setZoom(popupMap.getZoom() - 1);
    });

    // Geofence controls
    document.getElementById('popupGeofenceBtn').addEventListener('click', () => {
        popupIsGeofenceMode = !popupIsGeofenceMode;
        const saveBtn = document.getElementById('popupGeofenceSaveBtn');
        const cancelBtn = document.getElementById('popupGeofenceCancelBtn');
        if (popupIsGeofenceMode) {
            if (popupDrawingManager) popupDrawingManager.setDrawingMode(google.maps.drawing.OverlayType.POLYGON);
            if (saveBtn) saveBtn.style.display = 'inline-block';
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
        } else {
            if (popupDrawingManager) popupDrawingManager.setDrawingMode(null);
            if (saveBtn) saveBtn.style.display = 'none';
            if (cancelBtn) cancelBtn.style.display = 'none';
        }
    });

    document.getElementById('popupGeofenceCancelBtn').addEventListener('click', () => {
        if (popupDrawingManager) popupDrawingManager.setDrawingMode(null);
        popupIsGeofenceMode = false;
        const saveBtn = document.getElementById('popupGeofenceSaveBtn');
        const cancelBtn = document.getElementById('popupGeofenceCancelBtn');
        if (saveBtn) saveBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';
    });

    // Search functionality
    const searchInput = document.getElementById('popupMapSearchInput');
    const searchBtn = document.getElementById('popupMapSearchBtn');
    const searchClear = document.getElementById('popupMapSearchClear');

    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            popupSearchLocation();
        });
    }

    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                popupSearchLocation();
            }
        });

        searchInput.addEventListener('input', () => {
            if (searchInput.value.trim().length > 0) {
                if (searchClear) searchClear.style.display = 'inline-block';
            } else {
                if (searchClear) searchClear.style.display = 'none';
            }
        });
    }

    if (searchClear) {
        searchClear.addEventListener('click', () => {
            popupClearSearch();
        });
    }

    // Device location toggle
    const toggleLocationBtn = document.getElementById('popupToggleDeviceLocation');
    if (toggleLocationBtn) {
        toggleLocationBtn.addEventListener('click', () => {
            if (popupUseDeviceLocation && popupGeolocationWatchId !== null) {
                stopPopupDeviceLocationTracking();
                popupUseDeviceLocation = false;
                toggleLocationBtn.classList.remove('active');
            } else {
                startPopupDeviceLocationTracking();
                popupUseDeviceLocation = true;
                toggleLocationBtn.classList.add('active');
            }
        });
    }
}

// Geocoding functions
function popupSearchLocation() {
    const searchInput = document.getElementById('popupMapSearchInput');
    if (!searchInput || !popupGeocoder || !popupMap) {
        alert('Search functionality not available');
        return;
    }

    const address = searchInput.value.trim();
    if (!address) {
        alert('Please enter a location to search');
        return;
    }

    const searchBtn = document.getElementById('popupMapSearchBtn');
    if (searchBtn) {
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    }

    popupGeocoder.geocode({ address: address }, (results, status) => {
        if (searchBtn) {
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search"></i>';
        }

        if (status === 'OK' && results && results.length > 0) {
            const result = results[0];
            const location = result.geometry.location;
            const lat = location.lat();
            const lng = location.lng();

            popupMap.setCenter({ lat, lng });
            popupMap.setZoom(15);

            // Add search marker
            if (popupSearchMarker) {
                popupSearchMarker.setMap(null);
            }

            popupSearchMarker = new google.maps.Marker({
                position: { lat, lng },
                map: popupMap,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 10,
                    fillColor: '#FFA500',
                    fillOpacity: 1,
                    strokeColor: '#FFFFFF',
                    strokeWeight: 2
                },
                title: result.formatted_address,
                zIndex: 999
            });

            searchInput.value = result.formatted_address;
        } else if (status === 'ZERO_RESULTS') {
            alert('No results found for this location');
        } else {
            alert(`Geocoding failed: ${status}`);
        }
    });
}

function popupClearSearch() {
    const searchInput = document.getElementById('popupMapSearchInput');
    if (searchInput) searchInput.value = '';

    const searchClear = document.getElementById('popupMapSearchClear');
    if (searchClear) searchClear.style.display = 'none';

    if (popupSearchMarker) {
        popupSearchMarker.setMap(null);
        popupSearchMarker = null;
    }

    if (popupMap && popupCurrentDronePosition) {
        popupMap.setCenter({
            lat: popupCurrentDronePosition.lat,
            lng: popupCurrentDronePosition.lng
        });
    }
}

// Device location tracking functions
function startPopupDeviceLocationTracking() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by this browser');
        return;
    }

    const toggleBtn = document.getElementById('popupToggleDeviceLocation');
    if (toggleBtn) {
        toggleBtn.classList.add('active');
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
            popupHandleDeviceLocationUpdate(position);
        },
        (error) => {
            popupHandleGeolocationError(error);
            if (error.code === error.PERMISSION_DENIED && toggleBtn) {
                toggleBtn.classList.remove('active');
            }
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );

    popupGeolocationWatchId = navigator.geolocation.watchPosition(
        (position) => {
            popupHandleDeviceLocationUpdate(position);
        },
        (error) => {
            popupHandleGeolocationError(error);
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 5000
        }
    );

    console.log('Popup device location tracking started');
}

function popupHandleDeviceLocationUpdate(position) {
    const lat = position.coords.latitude;
    const lng = position.coords.longitude;
    const altitude = position.coords.altitude || 0;
    const heading = position.coords.heading || 0;

    // Calculate heading if not available
    let calculatedHeading = heading;
    if (!heading && popupCurrentDronePosition.lat && popupCurrentDronePosition.lng) {
        const prevLat = popupCurrentDronePosition.lat;
        const prevLng = popupCurrentDronePosition.lng;
        const dLng = (lng - prevLng) * Math.PI / 180;
        const prevLatRad = prevLat * Math.PI / 180;
        const latRad = lat * Math.PI / 180;
        
        const y = Math.sin(dLng) * Math.cos(latRad);
        const x = Math.cos(prevLatRad) * Math.sin(latRad) - 
                 Math.sin(prevLatRad) * Math.cos(latRad) * Math.cos(dLng);
        
        calculatedHeading = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
    }

    // Center map on first location
    if (!popupFirstLocationReceived && popupMap) {
        popupFirstLocationReceived = true;
        popupMap.setCenter({ lat, lng });
        popupMap.setZoom(17);
        console.log('Popup map centered on device location:', lat, lng);
    }

    updatePopupDronePosition({
        lat: lat,
        lng: lng,
        altitude: altitude,
        heading: calculatedHeading
    });
}

function popupHandleGeolocationError(error) {
    let errorMessage = 'Location tracking error: ';
    switch(error.code) {
        case error.PERMISSION_DENIED:
            errorMessage += 'Permission denied';
            alert('Location permission denied. Please allow location access.');
            break;
        case error.POSITION_UNAVAILABLE:
            errorMessage += 'Position unavailable';
            break;
        case error.TIMEOUT:
            errorMessage += 'Request timeout';
            break;
        default:
            errorMessage += 'Unknown error';
            break;
    }
    console.error(errorMessage, error);
}

function stopPopupDeviceLocationTracking() {
    if (popupGeolocationWatchId !== null) {
        navigator.geolocation.clearWatch(popupGeolocationWatchId);
        popupGeolocationWatchId = null;
        console.log('Popup device location tracking stopped');
        
        const toggleBtn = document.getElementById('popupToggleDeviceLocation');
        if (toggleBtn) {
            toggleBtn.classList.remove('active');
        }
    }
}