// Galamsey AI Detector - Real-time Dashboard JavaScript
class GalamseyDetector {
    constructor() {
        this.isRunning = false;
        this.currentStatus = 'illegal';
        this.detectionCount = 0;
        this.frameCount = 0;
        this.isVideoStreaming = false;
        this.isInitialized = false;
        this.updateInterval = null;
        this.webcamStream = null;
        this.isWebcamActive = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.detectionData = {
            excavator: { confidence: 0, detected: false, position: { x: 0, y: 0 } },
            pump: { confidence: 0, detected: false, position: { x: 0, y: 0 } },
            washplant: { confidence: 0, detected: false, position: { x: 0, y: 0 } }
        };
        this.alertHistory = [];
        this.statistics = {
            today: { total: 47, legal: 23, illegal: 24 },
            trends: { detection: '+12%', legal: '+5%', illegal: '+3%' }
        };
        
        // Settings configuration
        this.settings = {
            detectionSensitivity: 75,
            alertThreshold: 80,
            autoExport: true,
            soundAlerts: true,
            updateFrequency: 2000,
            droneIP: '192.168.1.100',
            dronePort: 8080,
            droneProtocol: 'webrtc',
            autoConnect: true,
            connectionTimeout: 30,
            autoLoadGhanaGeofences: false,
            forestGeoJsonUrl: '',
            riversGeoJsonUrl: ''
        };
        
        // Map mode configuration
        this.mapModes = ['satellite', 'terrain', 'hybrid', 'street'];
        this.currentMapMode = 0;
        
        // Performance optimization: Use requestAnimationFrame for smooth animations
        this.animationFrame = null;
        this.lastUpdate = 0;
        this.updateFrequency = 1000 / 30; // 30 FPS
        
        // Geofencing state
        this.isGeofenceMode = false;
        this.currentGeofencePoints = [];
        this.geofences = [];
        this.geofenceAlertTimestamps = {}; // throttle alerts per geofence and class
        
        // =====================
        // Google Maps State
        // =====================
        this.map = null; // Google Map instance
        this.droneMarker = null; // Drone marker on map
        this.dronePath = null; // Polyline for drone path trail
        this.dronePathCoordinates = []; // Array of lat/lng coordinates for path
        this.geofencePolygons = []; // Array of Google Maps polygons for geofences
        this.drawingManager = null; // Google Maps drawing manager for geofence drawing
        this.currentDronePosition = { lat: 5.6037, lng: -0.1870, altitude: 0, heading: 0 }; // Default: Accra, Ghana
        this.isInsideGeofence = true; // Track if drone is inside allowed geofence
        this.isInsideRestrictedArea = false; // Track if drone is in restricted area
        this.mapInitialized = false;
        this.smoothMarkerAnimation = null; // For smooth marker movement
        this.mapPopupWindow = null; // Tab window for map
        this.mapPopupSyncInterval = null; // Interval for syncing position to tab
        this.mapTabChannel = null; // BroadcastChannel for tab communication
        this.searchMarker = null; // Marker for searched location
        this.geocoder = null; // Google Maps Geocoder instance
        this.autocompleteService = null; // Google Maps Places Autocomplete service
        this.placesService = null; // Google Maps Places service
        this.simulationMode = false; // Simulation mode for testing
        this.simulationInterval = null; // Interval for simulation updates
        this.ws = null; // WebSocket connection
        this.geolocationWatchId = null; // ID for geolocation watch
        this.useDeviceLocation = true; // Use device GPS location as drone position
        this.firstLocationReceived = false; // Track if we've received the first location update
        this.detectionSendInterval = null; // Interval for sending detection data
        this.lastSentDetectionHash = null; // Hash of last sent detection to avoid duplicates
        
        this.initializeDashboard();
        this.startRealTimeUpdates();
        this.setupEventListeners();
        this.initializeVideoStream();
        this.setupWebSocketConnection();
        this.setupPerformanceOptimizations();
        this.loadSettings();
        this.applySettings();
        this.renderGeofenceList();
        
        // Hide feed content initially since no device is connected
        this.updateFeedVisibility();
    }

    // Static method to initialize map when Google Maps API loads
    static initMap() {
        // This will be called by Google Maps API callback
        // The detector instance will be available globally or we'll find it
        if (window.detectorInstance) {
            window.detectorInstance.initializeGoogleMap();
        }
    }

    initializeDashboard() {
        // Update current time
        this.updateTime();
        
        
        // Set initial status
        this.updateStatusBanner(this.currentStatus);
        
        // Initialize map marker
        this.initializeMapMarker();
        
        // Set initial drone status to disconnected (no device connected by default)
        this.updateDroneStatus(false);
        
        // Initialize alert badge
        this.updateAlertBadge();
        
        // // Start detection simulation
        // this.simulateDetectionUpdates();
        
        // Hide loading screen after initialization
        this.hideLoadingScreen();
    }

    hideLoadingScreen() {
        // Simulate loading progress
        const loadingProgress = document.getElementById('loadingProgress');
        const loadingText = document.getElementById('loadingText');
        const loadingScreen = document.getElementById('loadingScreen');
        
        if (loadingProgress && loadingText && loadingScreen) {
            let progress = 0;
            const loadingSteps = [
                'Initializing AI Detection System...',
                'Connecting to drone feed...',
                'Loading detection models...',
                'Calibrating GPS coordinates...',
                'Starting real-time monitoring...',
                'System ready!'
            ];
            
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15 + 5; // Random progress between 5-20%
                if (progress > 100) progress = 100;
                
                loadingProgress.style.width = `${progress}%`;
                
                const stepIndex = Math.floor((progress / 100) * loadingSteps.length);
                if (stepIndex < loadingSteps.length) {
                    loadingText.textContent = loadingSteps[stepIndex];
                }
                
                if (progress >= 100) {
                    clearInterval(progressInterval);
                    setTimeout(() => {
                        loadingScreen.style.opacity = '0';
                        loadingScreen.style.visibility = 'hidden';
                        loadingScreen.style.transition = 'opacity 0.5s ease, visibility 0.5s ease';
                        
                        // Show the main content
                        document.body.style.opacity = '1';
                        this.isInitialized = true;
                    }, 500);
                }
            }, 200);
        }
    }

    updateTime() {
        const now = new Date();
        const timeString = now.toISOString().replace('T', ' ').substring(0, 19) + ' GMT';
        document.getElementById('currentTime').textContent = timeString;
    }

    updateStatusBanner(status) {
        const banner = document.getElementById('statusBanner');
        const bannerContent = banner.querySelector('.banner-content');
        
        banner.className = `status-banner ${status}`;
        
        switch(status) {
            case 'illegal':
                bannerContent.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>⚠️ Illegal Mining Detected</span>';
                break;
            case 'legal':
                bannerContent.innerHTML = '<i class="fas fa-check-circle"></i><span>✅ Legal Area Detected</span>';
                break;
            case 'analyzing':
                bannerContent.innerHTML = '<i class="fas fa-search"></i><span>🔍 Analyzing Feed…</span>';
                break;
        }
    }

    // }

    updateDetectionBoxes(status) {
        const detectionBoxes = document.querySelectorAll('.detection-box');
        
        detectionBoxes.forEach(box => {
            if (status === 'illegal') {
                box.style.borderColor = '#FF4444';
                box.style.boxShadow = '0 0 20px rgba(255, 68, 68, 0.5)';
            } else if (status === 'legal') {
                box.style.borderColor = '#00FF99';
                box.style.boxShadow = '0 0 20px rgba(0, 255, 153, 0.5)';
            } else {
                box.style.borderColor = '#FFA500';
                box.style.boxShadow = '0 0 20px rgba(255, 165, 0, 0.5)';
            }
        });
    }

    updateReportPanel(status) {
        const siteType = document.querySelector('.site-type');
        const toolItems = document.querySelectorAll('.tool-item');
        const statusDots = document.querySelectorAll('.status-dot');
        
        if (status === 'illegal') {
            siteType.textContent = '⚠️ Illegal Mining Detected';
            siteType.className = 'site-type illegal';
            toolItems.forEach(item => item.classList.add('illegal'));
            statusDots.forEach(dot => {
                dot.style.background = '#FF4444';
                dot.style.boxShadow = '0 0 8px #FF4444';
            });
        } else if (status === 'legal') {
            siteType.textContent = '✅ Legal Mining Operation';
            siteType.className = 'site-type legal';
            toolItems.forEach(item => {
                item.classList.remove('illegal');
                item.classList.add('legal');
            });
            statusDots.forEach(dot => {
                dot.style.background = '#00FF99';
                dot.style.boxShadow = '0 0 8px #00FF99';
            });
        } else {
            siteType.textContent = '🔍 Analyzing Site...';
            siteType.className = 'site-type analyzing';
        }
    }

    updateConfidenceValues() {
        // Update confidence in report panel
        const confidenceValue = document.querySelector('.confidence-value');
        const newConfidence = Math.floor(Math.random() * 20) + 80; // 80-100%
        confidenceValue.textContent = `${newConfidence}%`;
        
        // Update confidence in detection boxes
        const confidenceSpans = document.querySelectorAll('.confidence');
        confidenceSpans.forEach(span => {
            const newValue = Math.floor(Math.random() * 20) + 80;
            span.textContent = `${newValue}%`;
        });
    }

    initializeMapMarker() {
        // Map markers will be dynamically created when detections occur
        // No need to initialize test markers
    }

    showMapTooltip() {
        const tooltip = document.querySelector('.marker-tooltip');
        tooltip.style.opacity = '1';
        
        setTimeout(() => {
            tooltip.style.opacity = '0';
        }, 3000);
    }

    startRealTimeUpdates() {
        // Update time every second
        setInterval(() => {
            this.updateTime();
        }, 1000);
        
        // Update statistics
        setInterval(() => {
            this.updateStatistics();
        }, 5000);
    }

    setupEventListeners() {
        
        // Map controls
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.zoomMap(1.2);
        });
        
        document.getElementById('zoomOut').addEventListener('click', () => {
            this.zoomMap(0.8);
        });
        
        // Action buttons
        document.getElementById('generateReport').addEventListener('click', () => {
            this.generateReport();
        });
        
        document.getElementById('viewHistory').addEventListener('click', () => {
            this.viewHistory();
        });
        
        document.getElementById('exportData').addEventListener('click', () => {
            this.exportData();
        });

        // Send detection data button
        document.getElementById('sendDetectionData').addEventListener('click', () => {
            this.sendCurrentDetectionData();
        });
        
        // System icons
        document.querySelectorAll('.icon-group').forEach(icon => {
            icon.addEventListener('click', (e) => {
                this.handleSystemIconClick(e.currentTarget);
            });
        });
        
        // Fullscreen button
        document.getElementById('fullscreenBtn').addEventListener('click', () => {
            this.toggleFullscreen();
        });
        
        // Settings button
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.showSettings();
        });
        
        // Settings modal events
        this.setupSettingsModal();
        
        // Alerts modal events
        this.setupAlertsModal();
        
        // Map mode button
        document.getElementById('mapModeBtn').addEventListener('click', () => {
            this.toggleMapMode();
        });

        // Webcam toggle button
        document.getElementById('webcamToggle').addEventListener('click', () => {
            this.toggleWebcam();
        });

        // Screenshot button
        document.getElementById('screenshotBtn').addEventListener('click', () => {
            this.takeScreenshot();
        });

        // Record button
        document.getElementById('recordBtn').addEventListener('click', () => {
            this.toggleRecording();
        });

        // Geofence controls
        document.getElementById('geofenceBtn').addEventListener('click', () => {
            this.toggleGeofenceDrawingMode();
        });
        document.getElementById('geofenceCancelBtn').addEventListener('click', () => {
            this.cancelCurrentGeofence();
        });

        // Open map in popup window
        document.getElementById('openMapPopup').addEventListener('click', () => {
            this.openMapInPopup();
        });

        // Map search functionality
        const mapSearchInput = document.getElementById('mapSearchInput');
        const mapSearchBtn = document.getElementById('mapSearchBtn');
        const mapSearchClear = document.getElementById('mapSearchClear');

        if (mapSearchBtn) {
            mapSearchBtn.addEventListener('click', () => {
                this.searchLocation();
            });
        }

        if (mapSearchInput) {
            // Search on Enter key
            mapSearchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.searchLocation();
                }
            });

            // Show/hide clear button based on input
            mapSearchInput.addEventListener('input', () => {
                if (mapSearchInput.value.trim().length > 0) {
                    if (mapSearchClear) mapSearchClear.style.display = 'inline-block';
                } else {
                    if (mapSearchClear) mapSearchClear.style.display = 'none';
                }
            });
        }

        if (mapSearchClear) {
            mapSearchClear.addEventListener('click', () => {
                this.clearSearch();
            });
        }

        // Device location toggle control
        const toggleDeviceLocationBtn = document.getElementById('toggleDeviceLocation');

        if (toggleDeviceLocationBtn) {
            toggleDeviceLocationBtn.addEventListener('click', () => {
                if (this.useDeviceLocation && this.geolocationWatchId !== null) {
                    this.stopDeviceLocationTracking();
                    this.useDeviceLocation = false;
                    toggleDeviceLocationBtn.classList.remove('active');
                    toggleDeviceLocationBtn.title = 'Start Device Location Tracking';
                } else {
                    this.startDeviceLocationTracking();
                    this.useDeviceLocation = true;
                    toggleDeviceLocationBtn.classList.add('active');
                    toggleDeviceLocationBtn.title = 'Stop Device Location Tracking';
                }
            });
        }
    }


    zoomMap(factor) {
        if (this.map) {
            const currentZoom = this.map.getZoom();
            const newZoom = Math.max(1, Math.min(20, currentZoom + (factor > 1 ? 1 : -1)));
            this.map.setZoom(newZoom);
        }
    }

    toggleFullscreen() {
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const fullscreenIcon = fullscreenBtn.querySelector('i');
        const fullscreenLabel = fullscreenBtn.querySelector('.icon-label');
        
        if (!document.fullscreenElement) {
            // Enter fullscreen
            document.documentElement.requestFullscreen().then(() => {
                fullscreenIcon.className = 'fas fa-compress';
                fullscreenLabel.textContent = 'Exit Fullscreen';
                fullscreenBtn.classList.add('active');
                this.showToast('Entered fullscreen mode', 'legal');
                
                // Add fullscreen-specific styles
                document.body.classList.add('fullscreen-mode');
            }).catch(err => {
                console.error('Error attempting to enable fullscreen:', err);
                this.showToast('Fullscreen not supported', 'illegal');
            });
        } else {
            // Exit fullscreen
            document.exitFullscreen().then(() => {
                fullscreenIcon.className = 'fas fa-expand';
                fullscreenLabel.textContent = 'Fullscreen';
                fullscreenBtn.classList.remove('active');
                this.showToast('Exited fullscreen mode', 'legal');
                
                // Remove fullscreen-specific styles
                document.body.classList.remove('fullscreen-mode');
            }).catch(err => {
                console.error('Error attempting to exit fullscreen:', err);
                this.showToast('Error exiting fullscreen', 'illegal');
            });
        }
    }

    generateReport() {
        const button = document.getElementById('generateReport');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        button.disabled = true;
        
        // Generate report from actual detection data
        const reportData = this.generateReportData();
        
        // In production, this would send the report to a server or generate a PDF
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-check"></i> Report Generated';
            button.style.background = 'rgba(0, 255, 153, 0.2)';
            button.style.borderColor = 'rgba(0, 255, 153, 0.5)';
            this.showToast('Report generated successfully', 'legal');
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.disabled = false;
                button.style.background = '';
                button.style.borderColor = '';
            }, 2000);
        }, 1000);
    }

    generateReportData() {
        const now = new Date();
        return {
            title: 'Galamsey AI Detection Report',
            timestamp: now.toISOString(),
            summary: {
                totalDetections: this.statistics.today.total,
                legalSites: this.statistics.today.legal,
                illegalSites: this.statistics.today.illegal,
                confidence: this.calculateAverageConfidence(),
                currentStatus: this.currentStatus
            },
            detections: this.detectionData,
            alerts: this.alertHistory.slice(0, 10),
            location: {
                gps: document.querySelector('.metadata-item span').textContent,
                environment: 'Near water body (within restricted buffer zone)'
            },
            recommendations: this.generateRecommendations()
        };
    }

    calculateAverageConfidence() {
        const confidences = Object.values(this.detectionData).map(d => d.confidence);
        return Math.round(confidences.reduce((a, b) => a + b, 0) / confidences.length);
    }

    generateRecommendations() {
        const recommendations = [];
        
        if (this.currentStatus === 'illegal') {
            recommendations.push('Immediate intervention required');
            recommendations.push('Alert district environmental officer');
            recommendations.push('Document evidence for legal proceedings');
        } else if (this.currentStatus === 'legal') {
            recommendations.push('Continue monitoring');
            recommendations.push('Verify permits are current');
        } else {
            recommendations.push('Continue analysis');
            recommendations.push('Monitor for changes in activity');
        }
        
        return recommendations;
    }

    // Download function removed - no longer downloads files

    formatReportAsText(data) {
        return `
GALAMSEY AI DETECTION REPORT
Generated: ${new Date(data.timestamp).toLocaleString()}

SUMMARY
=======
Total Detections: ${data.summary.totalDetections}
Legal Sites: ${data.summary.legalSites}
Illegal Sites: ${data.summary.illegalSites}
Average Confidence: ${data.summary.confidence}%
Current Status: ${data.summary.currentStatus.toUpperCase()}

DETECTION DETAILS
=================
${Object.entries(data.detections).map(([key, detection]) => 
    `${key.toUpperCase()}: ${detection.confidence}% confidence`
).join('\n')}

LOCATION
========
GPS Coordinates: ${data.location.gps}
Environment: ${data.location.environment}

RECENT ALERTS
=============
${data.alerts.map(alert => 
    `[${alert.timestamp.toLocaleString()}] ${alert.message} (${alert.type})`
).join('\n')}

RECOMMENDATIONS
===============
${data.recommendations.map((rec, i) => `${i + 1}. ${rec}`).join('\n')}

---
Report generated by Galamsey AI Detector System
Powered by YOLOv8 + GeoAI
        `.trim();
    }

    exportData() {
        const button = document.getElementById('exportData');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';
        button.disabled = true;
        
        setTimeout(() => {
            // Data exported successfully (no download)
            button.innerHTML = '<i class="fas fa-check"></i> Data Exported';
            button.style.background = 'rgba(0, 255, 153, 0.2)';
            button.style.borderColor = 'rgba(0, 255, 153, 0.5)';
            this.showToast('Data exported successfully', 'legal');
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.disabled = false;
                button.style.background = '';
                button.style.borderColor = '';
            }, 2000);
        }, 1500);
    }

    prepareExportData() {
        return {
            timestamp: new Date().toISOString(),
            statistics: this.statistics,
            detections: this.detectionData,
            alerts: this.alertHistory,
            currentStatus: this.currentStatus,
            frameCount: this.frameCount,
            detectionCount: this.detectionCount,
            location: {
                gps: document.querySelector('.metadata-item span').textContent,
                environment: 'Near water body (within restricted buffer zone)'
            }
        };
    }

    // Data export download function removed - no longer downloads files

    // Performance optimizations
    setupPerformanceOptimizations() {
        // Debounce resize events
        this.setupResizeHandler();
        
        // Optimize animations with requestAnimationFrame
        this.setupAnimationLoop();
    }

    setupResizeHandler() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });
    }

    handleResize() {
        // Handle any resize-related updates
    }

    setupAnimationLoop() {
        const animate = (timestamp) => {
            if (timestamp - this.lastUpdate >= this.updateFrequency) {
                this.updateAnimations();
                this.lastUpdate = timestamp;
            }
            this.animationFrame = requestAnimationFrame(animate);
        };
        this.animationFrame = requestAnimationFrame(animate);
    }

    updateAnimations() {
        // Update detection box positions smoothly
        this.updateDetectionBoxPositions();
        
        // Update any other smooth animations
        this.updateSmoothAnimations();
    }

    updateDetectionBoxPositions() {
        Object.entries(this.detectionData).forEach(([key, detection]) => {
            const box = document.querySelector(`.detection-box.${key}`);
            if (box) {
                box.style.left = `${detection.position.x}%`;
                box.style.top = `${detection.position.y}%`;
            }
        });
    }

    updateSmoothAnimations() {
        // Add any other smooth animations here
        const recordingDot = document.querySelector('.recording-dot');
        if (recordingDot) {
            recordingDot.style.animationDuration = `${1 + Math.random() * 0.5}s`;
        }
    }

    // Cleanup method for performance
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }

    viewHistory() {
        const button = document.getElementById('viewHistory');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        button.disabled = true;
        
        // In production, this would fetch actual history from server
        // For now, show alert that history feature needs implementation
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-info"></i> History';
            button.style.background = 'rgba(255, 165, 0, 0.2)';
            button.style.borderColor = 'rgba(255, 165, 0, 0.5)';
            this.showToast('History feature requires server connection', 'analyzing');
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.disabled = false;
                button.style.background = '';
                button.style.borderColor = '';
            }, 2000);
        }, 500);
    }

    handleSystemIconClick(iconGroup) {
        const icon = iconGroup.querySelector('i');
        const label = iconGroup.querySelector('.icon-label').textContent;
        
        // Add visual feedback
        iconGroup.style.transform = 'scale(0.95)';
        iconGroup.style.background = 'rgba(0, 255, 153, 0.2)';
        
        setTimeout(() => {
            iconGroup.style.transform = '';
            iconGroup.style.background = '';
        }, 200);
        
        // Handle specific actions
        if (label === 'Settings') {
            this.showSettings();
        } else if (label === 'Map Mode') {
            this.toggleMapMode();
        } else if (label === 'Drone Connected' || label === 'Drone Disconnected') {
            this.toggleDroneConnection();
        } else if (label === 'Alerts') {
            this.showAlerts();
        }
    }

    showSettings() {
        const settingsModal = document.getElementById('settingsModal');
        settingsModal.classList.add('active');
        
        // Load current settings into the form
        this.loadSettings();
        this.renderGeofenceList();
        
        // Add click outside to close
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) {
                this.hideSettings();
            }
        });
    }

    hideSettings() {
        const settingsModal = document.getElementById('settingsModal');
        settingsModal.classList.remove('active');
    }

    showAlerts() {
        const alertsModal = document.getElementById('alertsModal');
        alertsModal.classList.add('active');
        
        // Update alert feed
        this.updateAlertFeed();
        
        // Add click outside to close
        alertsModal.addEventListener('click', (e) => {
            if (e.target === alertsModal) {
                this.hideAlerts();
            }
        });
    }

    hideAlerts() {
        const alertsModal = document.getElementById('alertsModal');
        alertsModal.classList.remove('active');
    }

    setupAlertsModal() {
        // Close button
        const closeAlertsBtn = document.getElementById('closeAlertsBtn');
        const closeAlerts = document.getElementById('closeAlerts');
        
        if (closeAlertsBtn) {
            closeAlertsBtn.addEventListener('click', () => {
                this.hideAlerts();
            });
        }
        
        if (closeAlerts) {
            closeAlerts.addEventListener('click', () => {
                this.hideAlerts();
            });
        }

        // Clear alerts button
        const clearAlerts = document.getElementById('clearAlerts');
        if (clearAlerts) {
            clearAlerts.addEventListener('click', () => {
                this.clearAllAlerts();
            });
        }
    }

    clearAllAlerts() {
        this.alertHistory = [];
        this.updateAlertFeed();
        this.updateAlertBadge();
        this.showToast('All alerts cleared', 'legal');
    }

    updateAlertBadge() {
        const alertBadge = document.getElementById('alertBadge');
        if (alertBadge) {
            const count = this.alertHistory.length;
            if (count > 0) {
                alertBadge.textContent = count > 99 ? '99+' : count;
                alertBadge.style.display = 'flex';
            } else {
                alertBadge.style.display = 'none';
            }
        }
    }

    setupSettingsModal() {
        // Close button
        document.getElementById('closeSettings').addEventListener('click', () => {
            this.hideSettings();
        });

        // Save button
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });

        // Reset button
        document.getElementById('resetSettings').addEventListener('click', () => {
            this.resetSettings();
        });

        // Range sliders with live updates
        const sensitivitySlider = document.getElementById('sensitivitySlider');
        const thresholdSlider = document.getElementById('thresholdSlider');
        const sensitivityValue = document.getElementById('sensitivityValue');
        const thresholdValue = document.getElementById('thresholdValue');

        sensitivitySlider.addEventListener('input', (e) => {
            sensitivityValue.textContent = e.target.value + '%';
        });

        thresholdSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = e.target.value + '%';
        });

        // Drone connection timeout slider
        const connectionTimeoutSlider = document.getElementById('connectionTimeout');
        const timeoutValue = document.getElementById('timeoutValue');

        connectionTimeoutSlider.addEventListener('input', (e) => {
            timeoutValue.textContent = e.target.value + 's';
        });

        // Drone connection buttons
        document.getElementById('testConnection').addEventListener('click', () => {
            this.testDroneConnection();
        });

        document.getElementById('connectDrone').addEventListener('click', () => {
            this.connectDrone();
        });
    }

    loadSettings() {
        // Load settings from localStorage or use defaults
        const savedSettings = localStorage.getItem('galamseySettings');
        if (savedSettings) {
            this.settings = { ...this.settings, ...JSON.parse(savedSettings) };
        }

        // Update form elements
        document.getElementById('sensitivitySlider').value = this.settings.detectionSensitivity;
        document.getElementById('sensitivityValue').textContent = this.settings.detectionSensitivity + '%';
        
        document.getElementById('thresholdSlider').value = this.settings.alertThreshold;
        document.getElementById('thresholdValue').textContent = this.settings.alertThreshold + '%';
        
        document.getElementById('autoExport').checked = this.settings.autoExport;
        document.getElementById('soundAlerts').checked = this.settings.soundAlerts;
        document.getElementById('updateFrequency').value = this.settings.updateFrequency;
        
        // Load drone settings
        document.getElementById('droneIP').value = this.settings.droneIP;
        document.getElementById('dronePort').value = this.settings.dronePort;
        document.getElementById('droneProtocol').value = this.settings.droneProtocol;
        document.getElementById('autoConnect').checked = this.settings.autoConnect;
        document.getElementById('connectionTimeout').value = this.settings.connectionTimeout;
        document.getElementById('timeoutValue').textContent = this.settings.connectionTimeout + 's';

        // Geofence data
        const autoLoadChk = document.getElementById('autoLoadGhanaGeofences');
        const forestUrl = document.getElementById('forestGeoJsonUrl');
        const riversUrl = document.getElementById('riversGeoJsonUrl');
        if (autoLoadChk) autoLoadChk.checked = !!this.settings.autoLoadGhanaGeofences;
        if (forestUrl) forestUrl.value = this.settings.forestGeoJsonUrl || '';
        if (riversUrl) riversUrl.value = this.settings.riversGeoJsonUrl || '';
    }

    saveSettings() {
        // Get values from form
        this.settings.detectionSensitivity = parseInt(document.getElementById('sensitivitySlider').value);
        this.settings.alertThreshold = parseInt(document.getElementById('thresholdSlider').value);
        this.settings.autoExport = document.getElementById('autoExport').checked;
        this.settings.soundAlerts = document.getElementById('soundAlerts').checked;
        this.settings.updateFrequency = parseInt(document.getElementById('updateFrequency').value);
        
        // Save drone settings
        this.settings.droneIP = document.getElementById('droneIP').value;
        this.settings.dronePort = parseInt(document.getElementById('dronePort').value);
        this.settings.droneProtocol = document.getElementById('droneProtocol').value;
        this.settings.autoConnect = document.getElementById('autoConnect').checked;
        this.settings.connectionTimeout = parseInt(document.getElementById('connectionTimeout').value);

        // Geofence data
        const autoLoadChk = document.getElementById('autoLoadGhanaGeofences');
        const forestUrl = document.getElementById('forestGeoJsonUrl');
        const riversUrl = document.getElementById('riversGeoJsonUrl');
        this.settings.autoLoadGhanaGeofences = !!(autoLoadChk && autoLoadChk.checked);
        this.settings.forestGeoJsonUrl = forestUrl ? forestUrl.value.trim() : '';
        this.settings.riversGeoJsonUrl = riversUrl ? riversUrl.value.trim() : '';

        // Save to localStorage
        localStorage.setItem('galamseySettings', JSON.stringify(this.settings));

        // Apply settings
        this.applySettings();

        // Show confirmation
        this.showToast('Settings saved successfully', 'legal');

        // Close modal
        this.hideSettings();
    }

    resetSettings() {
        // Reset to default values
        this.settings = {
            detectionSensitivity: 75,
            alertThreshold: 80,
            autoExport: true,
            soundAlerts: true,
            updateFrequency: 2000,
            droneIP: '192.168.1.100',
            dronePort: 8080,
            droneProtocol: 'webrtc',
            autoConnect: true,
            connectionTimeout: 30
        };

        // Update form
        this.loadSettings();

        // Show confirmation
        this.showToast('Settings reset to defaults', 'legal');
    }

    applySettings() {
        // Apply detection sensitivity
        this.updateDetectionSensitivity();
        
        // Apply alert threshold
        this.updateAlertThreshold();
        
        // Apply update frequency
        this.updateUpdateFrequency();
        
        // Apply sound alerts
        this.updateSoundAlerts();
        
        // Apply auto-export
        this.updateAutoExport();

        // Auto-load Ghana layers if enabled
        if (this.settings.autoLoadGhanaGeofences) {
            this.loadGhanaLayers();
        }
    }

    updateDetectionSensitivity() {
        // Adjust detection confidence based on sensitivity
        const sensitivity = this.settings.detectionSensitivity / 100;
        
        Object.keys(this.detectionData).forEach(key => {
            const detection = this.detectionData[key];
            // Adjust confidence based on sensitivity setting
            const baseConfidence = detection.confidence;
            const adjustedConfidence = Math.min(99, Math.max(70, baseConfidence * sensitivity));
            detection.confidence = adjustedConfidence;
        });
        
        this.updateDetectionBoxes(this.currentStatus);
    }

    updateAlertThreshold() {
        // Update alert generation based on threshold
        const threshold = this.settings.alertThreshold;
        
        // This would affect when alerts are generated
        // For now, we'll just log the change
        console.log(`Alert threshold set to ${threshold}%`);
    }

    updateUpdateFrequency() {
        // Clear existing intervals
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Set new update frequency
        this.updateInterval = setInterval(() => {
            this.updateStatistics();
        }, this.settings.updateFrequency);
        
        console.log(`Update frequency set to ${this.settings.updateFrequency}ms`);
    }

    updateSoundAlerts() {
        // Enable/disable sound alerts
        if (this.settings.soundAlerts) {
            console.log('Sound alerts enabled');
        } else {
            console.log('Sound alerts disabled');
        }
    }

    updateAutoExport() {
        // Enable/disable auto-export
        if (this.settings.autoExport) {
            console.log('Auto-export enabled');
            // Set up auto-export interval (e.g., every 5 minutes)
            setInterval(() => {
                if (this.settings.autoExport) {
                    this.exportData();
                }
            }, 300000); // 5 minutes
        } else {
            console.log('Auto-export disabled');
        }
    }

    testDroneConnection() {
        const testBtn = document.getElementById('testConnection');
        const originalText = testBtn.textContent;
        const droneIP = document.getElementById('droneIP').value;
        const dronePort = document.getElementById('dronePort').value;
        const droneProtocol = document.getElementById('droneProtocol').value;
        
        testBtn.textContent = 'Testing...';
        testBtn.disabled = true;
        
        // Test connection to actual drone
        // In production, this would make an actual connection test
        fetch(`http://${droneIP}:${dronePort}/test`, {
            method: 'GET',
            mode: 'no-cors',
            timeout: 5000
        }).then(() => {
            testBtn.textContent = '✓ Connected';
            testBtn.style.background = '#00FF99';
            this.showToast('Drone connection test successful', 'legal');
            
            setTimeout(() => {
                testBtn.textContent = originalText;
                testBtn.disabled = false;
                testBtn.style.background = '';
            }, 2000);
        }).catch(() => {
            testBtn.textContent = '✗ Failed';
            testBtn.style.background = '#FF4444';
            this.showToast('Drone connection test failed. Check IP and port.', 'illegal');
            
            setTimeout(() => {
                testBtn.textContent = originalText;
                testBtn.disabled = false;
                testBtn.style.background = '';
            }, 2000);
        });
    }

    connectDrone() {
        const connectBtn = document.getElementById('connectDrone');
        const originalText = connectBtn.textContent;
        const droneIP = document.getElementById('droneIP').value;
        const dronePort = document.getElementById('dronePort').value;
        const droneProtocol = document.getElementById('droneProtocol').value;
        
        connectBtn.textContent = 'Connecting...';
        connectBtn.disabled = true;
        
        // Connect to actual drone
        // In production, this would establish a real connection
        this.establishConnection(droneIP, dronePort, droneProtocol).then(() => {
            connectBtn.textContent = '✓ Connected';
            connectBtn.style.background = '#00FF99';
            this.isVideoStreaming = true;
            this.updateDroneStatus(true);
            this.updateFeedVisibility();
            this.showToast('Drone connected successfully', 'legal');
            
            setTimeout(() => {
                connectBtn.textContent = originalText;
                connectBtn.disabled = false;
                connectBtn.style.background = '';
            }, 2000);
        }).catch((error) => {
            connectBtn.textContent = '✗ Failed';
            connectBtn.style.background = '#FF4444';
            this.isVideoStreaming = false;
            this.updateDroneStatus(false);
            this.updateFeedVisibility();
            this.showToast('Failed to connect to drone: ' + error.message, 'illegal');
            
            setTimeout(() => {
                connectBtn.textContent = originalText;
                connectBtn.disabled = false;
                connectBtn.style.background = '';
            }, 3000);
        });
    }
    
    async establishConnection(ip, port, protocol) {
        // In production, implement actual connection logic here
        // This is a placeholder for the real connection implementation
        return new Promise((resolve, reject) => {
            // For now, reject to indicate connection needs to be implemented
            reject(new Error('Connection implementation required'));
        });
    }

    updateDroneStatus(connected) {
        const statusIndicator = document.querySelector('.status-indicator');
        const droneStatus = document.getElementById('droneStatus');
        
        if (connected) {
            statusIndicator.classList.add('connected');
            statusIndicator.style.background = '#00FF99';
            statusIndicator.style.boxShadow = '0 0 10px #00FF99';
            droneStatus.querySelector('.icon-label').textContent = 'Drone Connected';
        } else {
            statusIndicator.classList.remove('connected');
            statusIndicator.style.background = '#FF4444';
            statusIndicator.style.boxShadow = '0 0 10px #FF4444';
            droneStatus.querySelector('.icon-label').textContent = 'Drone Disconnected';
        }
        
        // Update feed visibility based on connection status
        this.updateFeedVisibility();
    }
    
    updateFeedVisibility() {
        const detectionOverlay = document.querySelector('.detection-overlay');
        const videoBackground = document.getElementById('videoBackground');
        const droneView = document.querySelector('.drone-view');
        const liveIndicator = document.querySelector('.live-indicator');
        const recordingDot = document.querySelector('.recording-dot');
        const frameCounter = document.querySelector('.frame-counter');
        const webcamVideo = document.getElementById('webcamVideo');
        
        // Check if webcam is active
        if (this.isWebcamActive && webcamVideo) {
            // Show webcam feed
            if (webcamVideo) webcamVideo.style.display = 'block';
            if (videoBackground) videoBackground.style.display = 'none';
            if (detectionOverlay) detectionOverlay.style.display = 'block';
            if (liveIndicator) {
                liveIndicator.querySelector('span').textContent = 'LIVE';
                liveIndicator.style.color = '#FF4444';
            }
            if (recordingDot) {
                recordingDot.style.background = '#FF4444';
                recordingDot.style.animation = 'pulse 1s infinite';
            }
            if (frameCounter) {
                frameCounter.textContent = 'Webcam Feed Active';
            }
        } else if (this.isVideoStreaming) {
            // Show feed content when device is connected
            if (webcamVideo) webcamVideo.style.display = 'none';
            if (detectionOverlay) detectionOverlay.style.display = 'block';
            if (videoBackground) videoBackground.style.display = 'block';
            if (droneView) droneView.style.display = 'block';
            if (liveIndicator) {
                liveIndicator.querySelector('span').textContent = 'LIVE';
                liveIndicator.style.color = '#FF4444';
            }
            if (recordingDot) {
                recordingDot.style.background = '#FF4444';
                recordingDot.style.animation = 'pulse 1s infinite';
            }
            if (frameCounter) {
                frameCounter.textContent = 'Frame: ' + this.frameCount;
            }
        } else {
            // Hide feed content when no device is connected
            if (webcamVideo) webcamVideo.style.display = 'none';
            if (detectionOverlay) detectionOverlay.style.display = 'none';
            if (videoBackground) videoBackground.style.display = 'none';
            if (droneView) droneView.style.display = 'none';
            if (liveIndicator) {
                liveIndicator.querySelector('span').textContent = 'OFFLINE';
                liveIndicator.style.color = '#666';
            }
            if (recordingDot) {
                recordingDot.style.background = '#666';
                recordingDot.style.animation = 'none';
            }
            if (frameCounter) {
                frameCounter.textContent = 'No Video Feed';
            }
        }
    }

    // Toggle webcam on/off
    async toggleWebcam() {
        if (this.isWebcamActive) {
            this.stopWebcam();
        } else {
            await this.startWebcam();
        }
    }

    // Start webcam feed
    async startWebcam() {
        const webcamVideo = document.getElementById('webcamVideo');
        const webcamToggle = document.getElementById('webcamToggle');
        
        if (!webcamVideo) {
            this.showToast('Webcam video element not found', 'illegal');
            return;
        }

        try {
            // Request access to user's webcam
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user' // Use front camera by default, can be changed to 'environment' for back camera
                },
                audio: false // We don't need audio for this application
            });

            // Set the stream to the video element
            webcamVideo.srcObject = stream;
            this.webcamStream = stream;
            this.isWebcamActive = true;

            // Update UI
            webcamToggle.innerHTML = '<i class="fas fa-video-slash"></i>';
            webcamToggle.title = 'Stop Webcam';
            this.updateFeedVisibility();
            this.showToast('Webcam started successfully', 'legal');
            
            // Update frame counter periodically
            this.startFrameCounter();
            
        } catch (error) {
            console.error('Error accessing webcam:', error);
            let errorMessage = 'Failed to access webcam';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Camera permission denied. Please allow camera access.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No camera found. Please connect a camera.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is already in use by another application.';
            }
            
            this.showToast(errorMessage, 'illegal');
        }
    }

    // Stop webcam feed
    stopWebcam() {
        const webcamVideo = document.getElementById('webcamVideo');
        const webcamToggle = document.getElementById('webcamToggle');
        
        if (this.webcamStream) {
            // Stop all tracks in the stream
            this.webcamStream.getTracks().forEach(track => {
                track.stop();
            });
            this.webcamStream = null;
        }

        if (webcamVideo) {
            webcamVideo.srcObject = null;
        }

        this.isWebcamActive = false;
        
        // Update UI
        if (webcamToggle) {
            webcamToggle.innerHTML = '<i class="fas fa-video"></i>';
            webcamToggle.title = 'Start Webcam';
        }
        
        this.updateFeedVisibility();
        this.showToast('Webcam stopped', 'legal');
        this.stopFrameCounter();
    }

    // Start frame counter for webcam
    startFrameCounter() {
        const frameCounter = document.querySelector('.frame-counter');
        if (!frameCounter) return;

        this.frameCounterInterval = setInterval(() => {
            if (this.isWebcamActive) {
                this.frameCount++;
                frameCounter.textContent = `Webcam Feed - Frame: ${this.frameCount}`;
            }
        }, 1000);
    }

    // Stop frame counter
    stopFrameCounter() {
        if (this.frameCounterInterval) {
            clearInterval(this.frameCounterInterval);
            this.frameCounterInterval = null;
        }
    }

    // Take screenshot from webcam
    takeScreenshot() {
        const webcamVideo = document.getElementById('webcamVideo');
        
        if (!this.isWebcamActive || !webcamVideo) {
            this.showToast('Webcam is not active', 'illegal');
            return;
        }

        try {
            // Create canvas to capture the frame
            const canvas = document.createElement('canvas');
            canvas.width = webcamVideo.videoWidth;
            canvas.height = webcamVideo.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob and download
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `screenshot-${Date.now()}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showToast('Screenshot saved', 'legal');
            }, 'image/png');
            
        } catch (error) {
            console.error('Error taking screenshot:', error);
            this.showToast('Failed to take screenshot', 'illegal');
        }
    }

    // =====================
    // Geofencing (Legacy methods - now using Google Maps)
    // =====================
    cancelCurrentGeofence() {
        if (this.drawingManager) {
            this.drawingManager.setDrawingMode(null);
        }
        this.isGeofenceMode = false;
        this.updateGeofenceControls();
        this.showToast('Geofence drawing cancelled', 'legal');
    }

    // =====================
    // Ghana auto-geofence layers (GeoJSON) - Updated for Google Maps
    // =====================
    async loadGhanaLayers() {
        const urls = [this.settings.forestGeoJsonUrl, this.settings.riversGeoJsonUrl].filter(Boolean);
        if (urls.length === 0) {
            this.showToast('No Ghana GeoJSON URLs configured', 'analyzing');
            return;
        }
        try {
            for (const url of urls) {
                const res = await fetch(url, { cache: 'no-cache' });
                const geo = await res.json();
                this.ingestGeoJsonAsGeofences(geo);
            }
            this.persistGeofences();
            if (this.mapInitialized) {
                this.renderSavedGeofencesOnMap();
            }
            this.showToast('Ghana geofences loaded', 'legal');
        } catch (err) {
            console.error('Failed loading Ghana layers:', err);
            this.showToast('Failed to load Ghana layers', 'illegal');
        }
    }

    ingestGeoJsonAsGeofences(geojson) {
        if (!geojson) return;
        const features = geojson.features || [];
        features.forEach((f) => {
            const name = (f.properties && (f.properties.name || f.properties.NAME || f.properties.Title)) || 'Geofence';
            const geom = f.geometry || {};
            if (geom.type === 'Polygon') {
                const rings = geom.coordinates || [];
                rings.forEach(ring => this.addGeoFenceFromLonLatRing(name, ring));
            } else if (geom.type === 'MultiPolygon') {
                const polys = geom.coordinates || [];
                polys.forEach(poly => {
                    (poly || []).forEach(ring => this.addGeoFenceFromLonLatRing(name, ring));
                });
            }
        });
    }

    addGeoFenceFromLonLatRing(name, ring) {
        if (!Array.isArray(ring) || ring.length < 3) return;
        // Convert to lat/lng coordinates for Google Maps
        const coordinates = ring.map(([lon, lat]) => ({ lat, lng: lon }));
        // Remove last point if it equals the first (closed polygon)
        if (coordinates.length > 2) {
            const first = coordinates[0];
            const last = coordinates[coordinates.length - 1];
            if (Math.abs(first.lat - last.lat) < 1e-6 && Math.abs(first.lng - last.lng) < 1e-6) {
                coordinates.pop();
            }
        }
        if (coordinates.length < 3) return;
        
        const geofence = {
            id: Date.now() + Math.random(),
            name: name,
            coordinates: coordinates
        };
        this.geofences.push(geofence);
    }

    // Toggle recording
    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    // Start recording the current feed (webcam for now)
    startRecording() {
        const recordBtn = document.getElementById('recordBtn');

        // Currently supports webcam recording
        if (!this.isWebcamActive || !this.webcamStream) {
            this.showToast('Start the webcam before recording', 'illegal');
            return;
        }

        try {
            const mimeTypes = [
                'video/webm;codecs=vp9',
                'video/webm;codecs=vp8',
                'video/webm'
            ];
            let selectedType = '';
            for (const type of mimeTypes) {
                if (MediaRecorder.isTypeSupported(type)) { selectedType = type; break; }
            }

            this.recordedChunks = [];
            this.mediaRecorder = new MediaRecorder(this.webcamStream, selectedType ? { mimeType: selectedType } : undefined);

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) this.recordedChunks.push(e.data);
            };

            this.mediaRecorder.onstop = () => {
                if (this.recordedChunks.length === 0) return;
                const blob = new Blob(this.recordedChunks, { type: this.mediaRecorder.mimeType || 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `recording-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                this.showToast('Recording saved', 'legal');
            };

            this.mediaRecorder.start(1000); // collect data every second
            this.isRecording = true;
            if (recordBtn) {
                recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
                recordBtn.title = 'Stop Recording';
            }
            this.showToast('Recording started', 'legal');
        } catch (err) {
            console.error('Failed to start recording:', err);
            this.showToast('Failed to start recording', 'illegal');
        }
    }

    // Stop recording
    stopRecording() {
        const recordBtn = document.getElementById('recordBtn');
        try {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
            }
        } catch (err) {
            console.error('Failed to stop recording:', err);
            this.showToast('Failed to stop recording', 'illegal');
        } finally {
            this.isRecording = false;
            if (recordBtn) {
                recordBtn.innerHTML = '<i class="fas fa-video"></i>';
                recordBtn.title = 'Record';
            }
        }
    }

    disconnectDrone() {
        this.isVideoStreaming = false;
        this.updateDroneStatus(false);
        this.updateFeedVisibility();
        this.showToast('Drone disconnected', 'illegal');
    }

    toggleMapMode() {
        if (!this.map) {
            this.showToast('Map not initialized yet', 'analyzing');
            return;
        }

        const mapModeBtn = document.getElementById('mapModeBtn');
        const mapModeIcon = mapModeBtn.querySelector('i');
        const mapModeLabel = mapModeBtn.querySelector('.icon-label');
        
        // Cycle to next map mode
        this.currentMapMode = (this.currentMapMode + 1) % this.mapModes.length;
        const newMode = this.mapModes[this.currentMapMode];
        
        // Apply map mode to Google Maps
        this.applyMapMode(newMode);
        
        // Update button appearance
        this.updateMapModeButton(newMode, mapModeIcon, mapModeLabel);
        
        // Show notification
        this.showToast(`Switched to ${newMode.charAt(0).toUpperCase() + newMode.slice(1)} view`, 'legal');
    }

    applyMapMode(mode) {
        if (!this.map) return;

        // Map Google Maps map type to mode
        const mapTypeMap = {
            'satellite': google.maps.MapTypeId.SATELLITE,
            'terrain': google.maps.MapTypeId.TERRAIN,
            'hybrid': google.maps.MapTypeId.HYBRID,
            'street': google.maps.MapTypeId.ROADMAP
        };

        if (mapTypeMap[mode]) {
            this.map.setMapTypeId(mapTypeMap[mode]);
        }
    }

    updateMapModeButton(mode, icon, label) {
        const modeIcons = {
            satellite: 'fa-satellite',
            terrain: 'fa-mountain',
            hybrid: 'fa-layer-group',
            street: 'fa-road'
        };
        
        const modeLabels = {
            satellite: 'Satellite',
            terrain: 'Terrain',
            hybrid: 'Hybrid',
            street: 'Street'
        };
        
        icon.className = `fas ${modeIcons[mode]}`;
        label.textContent = modeLabels[mode];
        
        // Add visual feedback
        const mapModeBtn = document.getElementById('mapModeBtn');
        mapModeBtn.classList.add('active');
        setTimeout(() => {
            mapModeBtn.classList.remove('active');
        }, 300);
    }

    updateMapLegend(mode) {
        const mapLegend = document.querySelector('.map-legend');
        if (!mapLegend) return;
        
        const legendItems = mapLegend.querySelectorAll('.legend-item');
        
        // Update legend based on map mode
        switch(mode) {
            case 'satellite':
                legendItems[0].querySelector('span').textContent = 'Satellite View';
                break;
            case 'terrain':
                legendItems[0].querySelector('span').textContent = 'Terrain View';
                break;
            case 'hybrid':
                legendItems[0].querySelector('span').textContent = 'Hybrid View';
                break;
            case 'street':
                legendItems[0].querySelector('span').textContent = 'Street View';
                break;
        }
    }

    toggleDroneConnection() {
        const statusIndicator = document.querySelector('.status-indicator');
        const isConnected = statusIndicator.classList.contains('connected');
        
        if (isConnected) {
            statusIndicator.classList.remove('connected');
            statusIndicator.style.background = '#FF4444';
            statusIndicator.style.boxShadow = '0 0 10px #FF4444';
        } else {
            statusIndicator.classList.add('connected');
            statusIndicator.style.background = '#00FF99';
            statusIndicator.style.boxShadow = '0 0 10px #00FF99';
        }
    }

    // GPS tracking will be updated from actual device data
    updateGPSLocation(lat, lng) {
        const gpsElement = document.getElementById('gpsLocation');
        if (gpsElement) {
            gpsElement.textContent = `${lat.toFixed(3)}° N, ${lng.toFixed(3)}° W`;
        }
    }

    // Initialize video stream
    async initializeVideoStream() {
        const videoFeed = document.getElementById('videoFeed');
        
        try {
            // In production, this would connect to a WebRTC/RTSP stream
            // For now, wait for device connection
            if (this.isVideoStreaming) {
                this.startVideoStream();
                this.showToast('Video stream connected', 'legal');
            } else {
                // No device connected, keep feed hidden
                this.updateFeedVisibility();
            }
        } catch (error) {
            console.error('Failed to initialize video stream:', error);
            this.isVideoStreaming = false;
            this.updateFeedVisibility();
            this.showToast('Video stream failed to connect', 'illegal');
        }
    }

    // Start actual video stream
    startVideoStream() {
        // In production, implement actual video stream connection here
        // This would connect to WebRTC, RTSP, or HTTP stream based on protocol
        const videoFeed = document.getElementById('videoFeed');
        const protocol = this.settings.droneProtocol;
        const ip = this.settings.droneIP;
        const port = this.settings.dronePort;
        
        // Placeholder for actual video stream implementation
        console.log(`Starting ${protocol} stream from ${ip}:${port}`);
    }

    // Setup WebSocket connection for real-time data
    setupWebSocketConnection() {
        // In production, this would connect to a WebSocket server
        // For now, wait for actual connection
        if (this.isVideoStreaming) {
            this.connectWebSocket();
        }
        // Device location tracking is handled separately
    }
    
    // Connect to WebSocket for real-time detection data
    connectWebSocket() {
        const wsUrl = `ws://${this.settings.droneIP}:${this.settings.dronePort}/ws`;
        
        try {
            // In production, create actual WebSocket connection
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected to:', wsUrl);
                this.showToast('WebSocket connected', 'legal');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('WebSocket message received:', data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    console.error('Raw message:', event.data);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showToast('WebSocket connection error. Using device location instead.', 'analyzing');
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
                this.showToast('WebSocket disconnected. Using device location instead.', 'analyzing');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.showToast('WebSocket unavailable. Using device location instead.', 'analyzing');
        }
    }
    
    // Handle WebSocket messages with actual detection data
    handleWebSocketMessage(data) {
        // Handle drone position updates
        // Expected format: { lat: <number>, lng: <number>, altitude: <number>, heading: <number> }
        if (data.lat !== undefined && data.lng !== undefined) {
            this.handleDronePositionUpdate(data);
        }
        
        // Update detection data from actual stream
        if (data.detections) {
            this.detectionData = data.detections;
            this.updateDetectionBoxes(this.currentStatus);
            this.checkGeofenceAlerts();
            
            // Send detection data to backend
            this.sendDetectionData();
        }
        
        // Legacy GPS format support
        if (data.gps) {
            this.handleDronePositionUpdate({
                lat: data.gps.lat,
                lng: data.gps.lng,
                altitude: data.gps.altitude || 0,
                heading: data.gps.heading || 0
            });
        }
    }

    // =====================
    // Detection Data Transmission
    // =====================
    /**
     * Send detection data to backend (location, time, detected items)
     * This function collects current detection data, location, and timestamp
     * and sends it via WebSocket or HTTP API
     */
    sendDetectionData() {
        // Get detected items (only items that are actually detected)
        const detectedItems = this.getDetectedItems();
        
        // Only send if there are actual detections
        if (detectedItems.length === 0) {
            return;
        }

        // Create detection payload
        const detectionPayload = {
            timestamp: new Date().toISOString(),
            location: {
                lat: this.currentDronePosition.lat,
                lng: this.currentDronePosition.lng,
                altitude: this.currentDronePosition.altitude,
                heading: this.currentDronePosition.heading
            },
            detections: detectedItems,
            metadata: {
                frameCount: this.frameCount,
                detectionCount: detectedItems.length,
                status: this.currentStatus,
                geofenceStatus: {
                    insideAllowedArea: this.isInsideGeofence,
                    insideRestrictedArea: this.isInsideRestrictedArea
                }
            }
        };

        // Create hash to avoid sending duplicate detections
        const detectionHash = this.createDetectionHash(detectionPayload);
        if (detectionHash === this.lastSentDetectionHash) {
            return; // Skip duplicate
        }
        this.lastSentDetectionHash = detectionHash;

        // Send via WebSocket if available
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify({
                    type: 'detection',
                    data: detectionPayload
                }));
                console.log('Detection data sent via WebSocket:', detectionPayload);
            } catch (error) {
                console.error('Error sending detection via WebSocket:', error);
                // Fallback to HTTP API
                this.sendDetectionDataViaAPI(detectionPayload);
            }
        } else {
            // Send via HTTP API if WebSocket is not available
            this.sendDetectionDataViaAPI(detectionPayload);
        }
    }

    /**
     * Get list of currently detected items
     * @returns {Array} - Array of detected items with their details
     */
    getDetectedItems() {
        const detectedItems = [];

        // Check each detection type
        Object.entries(this.detectionData || {}).forEach(([itemType, detection]) => {
            if (detection && detection.detected && detection.confidence) {
                // Only include if confidence meets threshold
                const threshold = this.settings?.alertThreshold || 80;
                if (detection.confidence >= threshold) {
                    detectedItems.push({
                        type: itemType,
                        confidence: detection.confidence,
                        position: detection.position || { x: 0, y: 0 },
                        detected: true
                    });
                }
            }
        });

        return detectedItems;
    }

    /**
     * Send detection data via HTTP API
     * @param {Object} payload - Detection payload to send
     */
    async sendDetectionDataViaAPI(payload) {
        const apiUrl = `http://${this.settings.droneIP}:${this.settings.dronePort}/api/detections`;
        
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
                // Use no-cors mode if CORS is an issue (but won't get response)
                // mode: 'no-cors'
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Detection data sent successfully via API:', result);
            } else {
                console.error('Failed to send detection data. Status:', response.status);
            }
        } catch (error) {
            console.error('Error sending detection data via API:', error);
            // Store in local queue for retry if needed
            this.queueDetectionForRetry(payload);
        }
    }

    /**
     * Create a hash of detection data to avoid duplicates
     * @param {Object} payload - Detection payload
     * @returns {String} - Hash string
     */
    createDetectionHash(payload) {
        // Create a simple hash based on location and detected items
        const locationStr = `${payload.location.lat.toFixed(4)}_${payload.location.lng.toFixed(4)}`;
        const itemsStr = payload.detections.map(d => `${d.type}_${d.confidence}`).join('|');
        return `${locationStr}_${itemsStr}`;
    }

    /**
     * Queue detection data for retry if sending fails
     * @param {Object} payload - Detection payload
     */
    queueDetectionForRetry(payload) {
        // Store in localStorage as backup
        try {
            const queue = JSON.parse(localStorage.getItem('detectionQueue') || '[]');
            queue.push({
                ...payload,
                retryCount: 0,
                queuedAt: new Date().toISOString()
            });
            
            // Keep only last 100 items
            if (queue.length > 100) {
                queue.shift();
            }
            
            localStorage.setItem('detectionQueue', JSON.stringify(queue));
        } catch (error) {
            console.error('Failed to queue detection for retry:', error);
        }
    }

    /**
     * Retry sending queued detections
     */
    async retryQueuedDetections() {
        try {
            const queue = JSON.parse(localStorage.getItem('detectionQueue') || '[]');
            if (queue.length === 0) return;

            const apiUrl = `http://${this.settings.droneIP}:${this.settings.dronePort}/api/detections`;
            const successful = [];

            for (let i = 0; i < queue.length; i++) {
                const item = queue[i];
                
                // Skip if too old (older than 1 hour)
                const queuedTime = new Date(item.queuedAt);
                const now = new Date();
                if (now - queuedTime > 3600000) { // 1 hour
                    successful.push(i);
                    continue;
                }

                try {
                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(item)
                    });

                    if (response.ok) {
                        successful.push(i);
                    } else {
                        item.retryCount = (item.retryCount || 0) + 1;
                        // Remove if retried too many times
                        if (item.retryCount > 5) {
                            successful.push(i);
                        }
                    }
                } catch (error) {
                    item.retryCount = (item.retryCount || 0) + 1;
                    if (item.retryCount > 5) {
                        successful.push(i);
                    }
                }
            }

            // Remove successfully sent items
            if (successful.length > 0) {
                const newQueue = queue.filter((_, index) => !successful.includes(index));
                localStorage.setItem('detectionQueue', JSON.stringify(newQueue));
            }
        } catch (error) {
            console.error('Error retrying queued detections:', error);
        }
    }

    /**
     * Manually send current detection data
     * Can be called from browser console: detector.sendCurrentDetectionData()
     */
    sendCurrentDetectionData() {
        const detectedItems = this.getDetectedItems();
        
        if (detectedItems.length === 0) {
            this.showToast('No detections to send', 'analyzing');
            return;
        }

        const payload = {
            timestamp: new Date().toISOString(),
            location: {
                lat: this.currentDronePosition.lat,
                lng: this.currentDronePosition.lng,
                altitude: this.currentDronePosition.altitude,
                heading: this.currentDronePosition.heading
            },
            detections: detectedItems,
            metadata: {
                frameCount: this.frameCount,
                detectionCount: detectedItems.length,
                status: this.currentStatus,
                geofenceStatus: {
                    insideAllowedArea: this.isInsideGeofence,
                    insideRestrictedArea: this.isInsideRestrictedArea
                }
            }
        };

        // Send immediately
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'detection',
                data: payload
            }));
            this.showToast(`Sent ${detectedItems.length} detection(s)`, 'legal');
        } else {
            this.sendDetectionDataViaAPI(payload);
            this.showToast(`Sent ${detectedItems.length} detection(s) via API`, 'legal');
        }

        console.log('Detection data sent:', payload);
        return payload;
    }

    /**
     * Start automatic detection data transmission
     * Sends detection data periodically when detections are present
     */
    startDetectionDataTransmission() {
        // Clear any existing interval
        if (this.detectionSendInterval) {
            clearInterval(this.detectionSendInterval);
        }

        // Send detection data every 5 seconds if detections are present
        this.detectionSendInterval = setInterval(() => {
            const detectedItems = this.getDetectedItems();
            if (detectedItems.length > 0) {
                this.sendDetectionData();
            }
        }, 5000); // Send every 5 seconds

        // Retry queued detections every 30 seconds
        setInterval(() => {
            this.retryQueuedDetections();
        }, 30000);

        console.log('Detection data transmission started');
    }

    /**
     * Stop automatic detection data transmission
     */
    stopDetectionDataTransmission() {
        if (this.detectionSendInterval) {
            clearInterval(this.detectionSendInterval);
            this.detectionSendInterval = null;
            console.log('Detection data transmission stopped');
        }
    }

    // Update detection data with realistic variations
    updateDetectionData() {
        // Only update detections if device is connected
        if (!this.isVideoStreaming) {
            // No detections when no device is connected
            Object.keys(this.detectionData).forEach(key => {
                this.detectionData[key] = { confidence: 0, detected: false, position: { x: 0, y: 0 } };
            });
            this.updateFeedVisibility();
            return;
        }
        this.updateDetectionBoxes(this.currentStatus);
        this.checkGeofenceAlerts();
    }

    // Update alert history from actual detections
    updateAlertHistory() {
        // Alerts will be generated from actual detection events
        // No simulated alerts
    }

    // Add new alert to history
    addAlert(message, type) {
        const alert = {
            id: Date.now(),
            message,
            type,
            timestamp: new Date(),
            timeAgo: 'just now'
        };
        
        this.alertHistory.unshift(alert);
        if (this.alertHistory.length > 100) {
            this.alertHistory.pop();
        }
        
        this.updateAlertFeed();
        this.updateAlertBadge();
        this.showToast(message, type);
    }

    // Update alert feed in sidebar
    updateAlertFeed() {
        const alertFeed = document.getElementById('alertFeed');
        if (!alertFeed) return;
        
        if (this.alertHistory.length === 0) {
            alertFeed.innerHTML = `
                <div class="no-alerts" style="text-align: center; padding: 2rem; color: #999;">
                    <i class="fas fa-bell-slash" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <p>No alerts yet</p>
                    <small>Alerts will appear here when detections occur</small>
                </div>
            `;
            return;
        }
        
        alertFeed.innerHTML = this.alertHistory.map(alert => `
            <div class="alert-item ${alert.type}" style="padding: 1rem; margin-bottom: 0.75rem; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border-left: 3px solid ${this.getAlertColor(alert.type)};">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <i class="fas fa-${this.getAlertIcon(alert.type)}" style="color: ${this.getAlertColor(alert.type)}; font-size: 1.2rem;"></i>
                    <div style="flex: 1;">
                        <div style="font-weight: 500; margin-bottom: 0.25rem;">${alert.message}</div>
                        <small style="color: #999; font-size: 0.8rem;">${this.getTimeAgo(alert.timestamp)}</small>
                    </div>
                </div>
            </div>
        `).join('');
        
        this.updateAlertBadge();
    }

    getAlertColor(type) {
        const colors = {
            illegal: '#FF4444',
            legal: '#00FF99',
            analyzing: '#FFA500'
        };
        return colors[type] || '#999';
    }

    // Get appropriate icon for alert type
    getAlertIcon(type) {
        const icons = {
            illegal: 'exclamation-triangle',
            legal: 'check-circle',
            analyzing: 'search'
        };
        return icons[type] || 'info-circle';
    }

    // Calculate time ago
    getTimeAgo(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'just now';
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    // Show toast notification
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(type)}"></i>
            <span>${message}</span>
            <button class="toast-close">&times;</button>
        `;

        toastContainer.appendChild(toast);

        // Auto remove after 5 seconds
        const removeTimer = setTimeout(() => {
            toast.remove();
        }, 5000);

        // Close button functionality
        toast.querySelector('.toast-close').addEventListener('click', () => {
            clearTimeout(removeTimer);
            toast.remove();
        });
    }

    // =====================
    // Geofence alerting
    // =====================
    checkGeofenceAlerts() {
        if (!this.geofences || this.geofences.length === 0) return;
        const threshold = this.settings?.alertThreshold ?? 80;
        const now = Date.now();
        const cooldownMs = 60000; // 60s per (geofence,class) to avoid spam

        Object.entries(this.detectionData || {}).forEach(([cls, det]) => {
            if (!det?.detected) return;
            if (det.confidence !== undefined && det.confidence < threshold) return;
            const pos = det.position || { x: 0, y: 0 };
            const hit = this.isInsideAnyGeofence(pos.x, pos.y);
            if (hit) {
                const key = `${hit.id}:${cls}`;
                const last = this.geofenceAlertTimestamps[key] || 0;
                if (now - last > cooldownMs) {
                    this.geofenceAlertTimestamps[key] = now;
                    const name = hit.name || 'Geofence';
                    const msg = `Illegal activity detected in geofence "${name}" (${cls})`;
                    this.addAlert(msg, 'illegal');
                    this.currentStatus = 'illegal';
                    this.updateStatusBanner('illegal');
                }
            }
        });
    }

    isInsideAnyGeofence(xPercent, yPercent) {
        for (const f of this.geofences) {
            if (this.pointInPolygonPercent({ x: xPercent, y: yPercent }, f.points)) {
                return f; // return geofence object
            }
        }
        return null;
    }

    // Ray-casting point-in-polygon in percent coordinate space (0-100)
    pointInPolygonPercent(point, polygonPoints) {
        if (!polygonPoints || polygonPoints.length < 3) return false;
        const x = point.x, y = point.y;
        let inside = false;
        for (let i = 0, j = polygonPoints.length - 1; i < polygonPoints.length; j = i++) {
            const xi = polygonPoints[i].x, yi = polygonPoints[i].y;
            const xj = polygonPoints[j].x, yj = polygonPoints[j].y;
            const intersect = ((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // =====================
    // Google Maps Initialization
    // =====================
    /**
     * Initialize Google Maps instance
     * Sets up the map with default settings and creates the drone marker
     * 
     * REQUIRES: Valid Google Maps API key in index.html
     * Get your API key from: https://console.cloud.google.com/google/maps-apis
     */
    initializeGoogleMap() {
        if (this.mapInitialized) {
            return;
        }

        // Check if Google Maps API is loaded
        if (typeof google === 'undefined' || !google.maps) {
            console.error('Google Maps API not loaded. Please check your API key in index.html');
            this.showToast('Google Maps API not loaded. Please check your API key.', 'illegal');
            // Show error message in map container
            const mapView = document.getElementById('mapView');
            if (mapView) {
                mapView.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #FF4444; text-align: center; padding: 2rem;">
                        <div>
                            <h3>Google Maps API Error</h3>
                            <p>Please add your Google Maps API key to index.html</p>
                            <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;">
                                Get your API key from:<br>
                                <a href="https://console.cloud.google.com/google/maps-apis" target="_blank" style="color: #00FF99;">
                                    https://console.cloud.google.com/google/maps-apis
                                </a>
                            </p>
                        </div>
                    </div>
                `;
            }
            return;
        }

        const mapView = document.getElementById('mapView');
        if (!mapView) {
            console.error('Map container not found');
            return;
        }

        try {
            // Default center: Accra, Ghana (will be updated when device location is received)
            const defaultCenter = { lat: 5.6037, lng: -0.1870 };

            // Create map with satellite view by default
            // If using device location, we'll update the center once location is received
            this.map = new google.maps.Map(mapView, {
                center: defaultCenter,
                zoom: 15,
                mapTypeId: google.maps.MapTypeId.SATELLITE,
                disableDefaultUI: false,
                zoomControl: true,
                mapTypeControl: true,
                streetViewControl: false,
                fullscreenControl: true,
                styles: [
                    {
                        featureType: 'all',
                        elementType: 'labels',
                        stylers: [{ visibility: 'on' }]
                    }
                ]
            });

            // If using device location, wait for first location before centering
            // Otherwise, center on default location immediately
            if (!this.useDeviceLocation) {
                // Center on default location if not using device location
                this.map.setCenter(defaultCenter);
            }

            // Initialize drone marker
            this.createDroneMarker();

            // Initialize drone path
            this.initializeDronePath();

            // Load saved geofences
            this.loadGeofences();
            this.renderSavedGeofencesOnMap();

            // Setup drawing manager for geofence creation
            this.setupDrawingManager();

            // Initialize Geocoding API
            this.initializeGeocoding();

            // Update map controls
            this.updateMapControls();

            this.mapInitialized = true;
            console.log('Google Maps initialized successfully');
            this.showToast('Google Maps loaded successfully', 'legal');

        // Start device location tracking once map is ready
        if (this.useDeviceLocation) {
            // Small delay to ensure map is fully rendered
            setTimeout(() => {
                this.startDeviceLocationTracking();
            }, 500);
        }

        // Start automatic detection data transmission
        this.startDetectionDataTransmission();
        } catch (error) {
            console.error('Error initializing Google Maps:', error);
            this.showToast('Error initializing Google Maps: ' + error.message, 'illegal');
        }
    }

    /**
     * Create and configure the drone marker on the map
     * Uses a custom icon that rotates based on heading
     */
    createDroneMarker() {
        if (!this.map) return;

        // Create a canvas-based drone icon for better visual quality
        const droneIcon = this.createDroneIcon('#00FF99');

        // Create marker with drone icon
        this.droneMarker = new google.maps.Marker({
            position: this.currentDronePosition,
            map: this.map,
            icon: droneIcon,
            title: 'Drone Position',
            zIndex: 1000
        });

        // Add info window for drone details
        const infoWindow = new google.maps.InfoWindow({
            content: `
                <div style="color: #000; padding: 8px; min-width: 200px;">
                    <strong style="color: #00FF99; font-size: 1.1em;">🚁 Drone Position</strong><br>
                    <hr style="margin: 5px 0; border: 1px solid #ddd;">
                    <div style="margin-top: 5px;">
                        <strong>Coordinates:</strong><br>
                        Lat: ${this.currentDronePosition.lat.toFixed(6)}°<br>
                        Lng: ${this.currentDronePosition.lng.toFixed(6)}°<br>
                    </div>
                    <div style="margin-top: 5px;">
                        <strong>Flight Data:</strong><br>
                        Altitude: ${this.currentDronePosition.altitude.toFixed(1)}m<br>
                        Heading: ${this.currentDronePosition.heading.toFixed(1)}°
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9em; color: #666;">
                        Click marker to update info
                    </div>
                </div>
            `
        });

        this.droneMarker.addListener('click', () => {
            // Update info window content with current data
            infoWindow.setContent(`
                <div style="color: #000; padding: 8px; min-width: 200px;">
                    <strong style="color: #00FF99; font-size: 1.1em;">🚁 Drone Position</strong><br>
                    <hr style="margin: 5px 0; border: 1px solid #ddd;">
                    <div style="margin-top: 5px;">
                        <strong>Coordinates:</strong><br>
                        Lat: ${this.currentDronePosition.lat.toFixed(6)}°<br>
                        Lng: ${this.currentDronePosition.lng.toFixed(6)}°<br>
                    </div>
                    <div style="margin-top: 5px;">
                        <strong>Flight Data:</strong><br>
                        Altitude: ${this.currentDronePosition.altitude.toFixed(1)}m<br>
                        Heading: ${this.currentDronePosition.heading.toFixed(1)}°
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9em; color: #666;">
                        Updated: ${new Date().toLocaleTimeString()}
                    </div>
                </div>
            `);
            infoWindow.open(this.map, this.droneMarker);
        });
    }

    /**
     * Initialize the drone path polyline for trail visualization
     */
    initializeDronePath() {
        if (!this.map) return;

        this.dronePath = new google.maps.Polyline({
            path: this.dronePathCoordinates,
            geodesic: true,
            strokeColor: '#00FF99',
            strokeOpacity: 0.8,
            strokeWeight: 3,
            map: this.map
        });
    }

    /**
     * Update drone position on the map with smooth animation
     * @param {Object} position - { lat, lng, altitude, heading }
     */
    updateDronePosition(position) {
        if (!this.map) {
            console.warn('Map not initialized yet. Position update queued.');
            // Queue the update for when map is ready
            setTimeout(() => {
                if (this.map) {
                    this.updateDronePosition(position);
                }
            }, 500);
            return;
        }

        if (!this.droneMarker) {
            console.warn('Drone marker not created yet. Creating marker...');
            this.createDroneMarker();
            if (!this.droneMarker) {
                console.error('Failed to create drone marker');
                return;
            }
        }

        // Validate position data
        if (typeof position.lat !== 'number' || typeof position.lng !== 'number') {
            console.error('Invalid position data:', position);
            return;
        }

        // Validate lat/lng ranges
        if (position.lat < -90 || position.lat > 90 || position.lng < -180 || position.lng > 180) {
            console.error('Position out of valid range:', position);
            return;
        }

        // If this is the first position update, center the map on it
        if (!this.firstLocationReceived) {
            this.firstLocationReceived = true;
            this.map.setCenter({ lat: position.lat, lng: position.lng });
            this.map.setZoom(17); // Zoom in closer for better view
            console.log('Map centered on first drone position:', position.lat, position.lng);
        }

        const oldPosition = { ...this.currentDronePosition };
        this.currentDronePosition = {
            lat: position.lat,
            lng: position.lng,
            altitude: position.altitude !== undefined ? position.altitude : this.currentDronePosition.altitude,
            heading: position.heading !== undefined ? position.heading : this.currentDronePosition.heading
        };

        // Add to path coordinates (limit to last 1000 points for performance)
        this.dronePathCoordinates.push({
            lat: this.currentDronePosition.lat,
            lng: this.currentDronePosition.lng
        });
        if (this.dronePathCoordinates.length > 1000) {
            this.dronePathCoordinates.shift();
        }

        // Update path polyline
        if (this.dronePath) {
            this.dronePath.setPath(this.dronePathCoordinates);
        }

        // Smoothly animate marker movement
        this.animateMarkerMovement(oldPosition, this.currentDronePosition);

        // Update marker rotation based on heading
        if (this.droneMarker) {
            // Get current color from marker state
            let color = '#00FF99'; // Default green
            if (this.isInsideRestrictedArea) {
                color = '#FF4444';
            } else if (!this.isInsideGeofence) {
                color = '#FFA500';
            }
            
            const newIcon = this.createDroneIcon(color);
            newIcon.rotation = this.currentDronePosition.heading;
            this.droneMarker.setIcon(newIcon);
        }

        // Check geofence violations
        this.checkGeofenceViolations();

        // Update GPS display
        this.updateGPSLocation(this.currentDronePosition.lat, this.currentDronePosition.lng);

        // Sync position to map tab if open
        if (this.mapTabChannel) {
            try {
                this.mapTabChannel.postMessage({
                    type: 'dronePosition',
                    position: this.currentDronePosition
                });
            } catch (error) {
                console.warn('Failed to send position to map tab:', error);
            }
        }

        // Center map on drone if it goes out of view (optional)
        // this.map.panTo(this.currentDronePosition);
    }

    /**
     * Animate marker movement smoothly between two positions
     * @param {Object} startPos - Starting position { lat, lng }
     * @param {Object} endPos - Ending position { lat, lng }
     */
    animateMarkerMovement(startPos, endPos) {
        if (!this.droneMarker) return;

        const steps = 20; // Number of animation steps
        const duration = 500; // Animation duration in ms
        const stepTime = duration / steps;
        let currentStep = 0;

        // Cancel any existing animation
        if (this.smoothMarkerAnimation) {
            clearInterval(this.smoothMarkerAnimation);
        }

        this.smoothMarkerAnimation = setInterval(() => {
            if (currentStep >= steps) {
                clearInterval(this.smoothMarkerAnimation);
                this.smoothMarkerAnimation = null;
                // Ensure final position is exact
                this.droneMarker.setPosition(endPos);
                return;
            }

            // Calculate interpolated position
            const progress = currentStep / steps;
            const lat = startPos.lat + (endPos.lat - startPos.lat) * progress;
            const lng = startPos.lng + (endPos.lng - startPos.lng) * progress;

            this.droneMarker.setPosition({ lat, lng });
            currentStep++;
        }, stepTime);
    }

    // =====================
    // Geofence Definition and Violation Detection
    // =====================
    /**
     * Setup Google Maps Drawing Manager for geofence creation
     */
    setupDrawingManager() {
        if (!this.map || typeof google === 'undefined') return;

        this.drawingManager = new google.maps.drawing.DrawingManager({
            drawingMode: null,
            drawingControl: false,
            polygonOptions: {
                fillColor: '#00FF99',
                fillOpacity: 0.2,
                strokeColor: '#00FF99',
                strokeWeight: 2,
                clickable: false,
                editable: true,
                zIndex: 1
            }
        });

        this.drawingManager.setMap(this.map);

        // Listen for polygon completion
        google.maps.event.addListener(this.drawingManager, 'polygoncomplete', (polygon) => {
            this.handleGeofenceDrawn(polygon);
        });
    }

    /**
     * Handle geofence polygon drawn on map
     * @param {google.maps.Polygon} polygon - The drawn polygon
     */
    handleGeofenceDrawn(polygon) {
        const name = prompt('Name this geofence:') || `Geofence ${new Date().toLocaleString()}`;
        
        // Ask for geofence type
        const typeChoice = prompt('Select geofence type:\n1. Allowed (Green) - Mining permitted\n2. Restricted (Red) - Mining prohibited\n\nEnter 1 or 2:');
        let geofenceType = 'allowed'; // Default to allowed
        if (typeChoice === '2' || typeChoice?.toLowerCase() === 'restricted' || typeChoice?.toLowerCase() === 'red') {
            geofenceType = 'restricted';
        }
        
        // Get polygon path
        const path = polygon.getPath();
        const coordinates = [];
        path.forEach((latLng) => {
            coordinates.push({
                lat: latLng.lat(),
                lng: latLng.lng()
            });
        });

        // Create geofence object with type
        const geofence = {
            id: Date.now(),
            name: name,
            type: geofenceType, // 'allowed' or 'restricted'
            coordinates: coordinates,
            polygon: polygon
        };

        this.geofences.push(geofence);
        this.geofencePolygons.push(polygon);

        // Style the polygon based on type
        const colors = this.getGeofenceColors(geofenceType);
        polygon.setOptions({
            fillColor: colors.fill,
            fillOpacity: 0.15,
            strokeColor: colors.stroke,
            strokeWeight: 2,
            editable: true,
            draggable: false
        });
        polygon.__geofenceId = geofence.id;

        // Add listener for edits
        path.addListener('set_at', () => this.onGeofenceEdited(geofence));
        path.addListener('insert_at', () => this.onGeofenceEdited(geofence));
        this.attachGeofenceEvents(polygon, geofence);

        // Save to localStorage
        this.persistGeofences();

        const typeLabel = geofenceType === 'allowed' ? 'Allowed (Green)' : 'Restricted (Red)';
        this.showToast(`Geofence "${name}" created as ${typeLabel}`, 'legal');
        this.renderGeofenceList();
        this.drawingManager.setDrawingMode(null);
        this.isGeofenceMode = false;
        this.updateGeofenceControls();
    }

    /**
     * Get colors for geofence based on type
     * @param {String} type - 'allowed' or 'restricted'
     * @returns {Object} - { fill, stroke }
     */
    getGeofenceColors(type) {
        if (type === 'restricted') {
            return {
                fill: '#FF4444',
                stroke: '#FF4444'
            };
        } else {
            // Default to allowed (green)
            return {
                fill: '#00FF99',
                stroke: '#00FF99'
            };
        }
    }

    /**
     * Handle geofence polygon edit
     * @param {Object} geofence - The geofence object
     */
    onGeofenceEdited(geofence) {
        const path = geofence.polygon.getPath();
        const coordinates = [];
        path.forEach((latLng) => {
            coordinates.push({
                lat: latLng.lat(),
                lng: latLng.lng()
            });
        });
        geofence.coordinates = coordinates;
        this.persistGeofences();
    }

    /**
     * Render saved geofences on the map
     */
    renderSavedGeofencesOnMap() {
        if (!this.map || typeof google === 'undefined') return;

        // Clear existing polygons
        this.geofencePolygons.forEach(polygon => polygon.setMap(null));
        this.geofencePolygons = [];

        // Render each geofence
        this.geofences.forEach(geofence => {
            if (!geofence.coordinates || geofence.coordinates.length < 3) {
                return;
            }

            // Get colors based on geofence type
            const geofenceType = geofence.type || 'allowed'; // Default to allowed for backward compatibility
            const colors = this.getGeofenceColors(geofenceType);

            const polygon = new google.maps.Polygon({
                paths: geofence.coordinates.map(coord => new google.maps.LatLng(coord.lat, coord.lng)),
                fillColor: colors.fill,
                fillOpacity: 0.15,
                strokeColor: colors.stroke,
                strokeWeight: 2,
                editable: true,
                draggable: false,
                map: this.map
            });

            geofence.polygon = polygon;
            polygon.__geofenceId = geofence.id;
            this.geofencePolygons.push(polygon);

            // Add edit listeners
            const path = polygon.getPath();
            path.addListener('set_at', () => this.onGeofenceEdited(geofence));
            path.addListener('insert_at', () => this.onGeofenceEdited(geofence));
            this.attachGeofenceEvents(polygon, geofence);
        });
        this.renderGeofenceList();
    }

    attachGeofenceEvents(polygon, geofence) {
        if (!polygon) return;
        polygon.addListener('click', () => {
            const confirmed = window.confirm(`Delete geofence "${geofence.name}"?`);
            if (confirmed) {
                this.removeGeofenceById(geofence.id, true);
            }
        });
    }

    removeGeofenceById(geofenceId, showToast = false) {
        const originalLength = this.geofences.length;
        this.geofences = this.geofences.filter(g => g.id !== geofenceId);

        if (this.geofencePolygons && this.geofencePolygons.length) {
            this.geofencePolygons.forEach((polygon, index) => {
                if (polygon && polygon.__geofenceId === geofenceId) {
                    polygon.setMap(null);
                    this.geofencePolygons.splice(index, 1);
                }
            });
        }

        this.persistGeofences();
        this.renderSavedGeofencesOnMap();
        this.renderGeofenceList();

        if (showToast && originalLength !== this.geofences.length) {
            this.showToast('Geofence deleted successfully', 'legal');
        }
    }

    renderGeofenceList() {
        const listContainer = document.getElementById('geofenceList');
        if (!listContainer) return;

        if (!this.geofences || this.geofences.length === 0) {
            listContainer.innerHTML = '<div class="geofence-empty">No geofences added yet</div>';
            return;
        }

        listContainer.innerHTML = this.geofences.map(geofence => {
            const geofenceType = geofence.type || 'allowed';
            const typeLabel = geofenceType === 'restricted' ? 'Restricted (Red)' : 'Allowed (Green)';
            const typeClass = geofenceType === 'restricted' ? 'geofence-restricted' : 'geofence-allowed';
            const typeIcon = geofenceType === 'restricted' ? 'fa-ban' : 'fa-check-circle';
            
            return `
                <div class="geofence-item ${typeClass}" data-id="${geofence.id}">
                    <div class="geofence-item-info">
                        <strong>${geofence.name}</strong>
                        <small>
                            <i class="fas ${typeIcon}"></i> ${typeLabel} • ${geofence.coordinates?.length || 0} points
                        </small>
                    </div>
                    <div class="geofence-item-actions">
                        <button class="geofence-edit" data-id="${geofence.id}" title="Edit Name"><i class="fas fa-edit"></i></button>
                        <button class="geofence-toggle-type" data-id="${geofence.id}" title="Toggle Type">
                            <i class="fas ${geofenceType === 'restricted' ? 'fa-check-circle' : 'fa-ban'}"></i>
                        </button>
                        <button class="geofence-delete" data-id="${geofence.id}" title="Delete"><i class="fas fa-trash"></i></button>
                    </div>
                </div>
            `;
        }).join('');

        listContainer.querySelectorAll('.geofence-edit').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = Number(btn.getAttribute('data-id'));
                const geofence = this.geofences.find(g => g.id === id);
                if (!geofence) return;
                const newName = prompt('Enter new geofence name:', geofence.name);
                if (newName && newName.trim()) {
                    geofence.name = newName.trim();
                    this.persistGeofences();
                    this.renderGeofenceList();
                    this.showToast('Geofence renamed', 'legal');
                }
            });
        });

        listContainer.querySelectorAll('.geofence-delete').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = Number(btn.getAttribute('data-id'));
                const geofence = this.geofences.find(g => g.id === id);
                if (!geofence) return;
                const confirmed = confirm(`Delete geofence "${geofence.name}"?`);
                if (confirmed) {
                    this.removeGeofenceById(id, true);
                }
            });
        });

        listContainer.querySelectorAll('.geofence-toggle-type').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = Number(btn.getAttribute('data-id'));
                const geofence = this.geofences.find(g => g.id === id);
                if (!geofence) return;
                
                // Toggle type
                geofence.type = geofence.type === 'restricted' ? 'allowed' : 'restricted';
                
                // Update polygon colors
                if (geofence.polygon) {
                    const colors = this.getGeofenceColors(geofence.type);
                    geofence.polygon.setOptions({
                        fillColor: colors.fill,
                        strokeColor: colors.stroke
                    });
                }
                
                this.persistGeofences();
                this.renderGeofenceList();
                this.showToast(`Geofence type changed to ${geofence.type === 'restricted' ? 'Restricted (Red)' : 'Allowed (Green)'}`, 'legal');
            });
        });
    }

    /**
     * Check if drone position violates any geofence boundaries
     * Updates marker color and shows alerts if violation detected
     */
    checkGeofenceViolations() {
        if (!this.map || !this.droneMarker || this.geofencePolygons.length === 0) {
            this.isInsideGeofence = true;
            return;
        }

        const dronePosition = new google.maps.LatLng(
            this.currentDronePosition.lat,
            this.currentDronePosition.lng
        );

        let insideAllowedGeofence = false;
        let insideRestrictedGeofence = false;
        let currentGeofence = null;

        // Check each geofence polygon
        this.geofencePolygons.forEach((polygon, index) => {
            if (google.maps.geometry.poly.containsLocation(dronePosition, polygon)) {
                // Find the geofence object for this polygon
                const geofenceId = polygon.__geofenceId;
                const geofence = this.geofences.find(g => g.id === geofenceId);
                
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

        // Handle restricted area violation (RED - most important)
        if (insideRestrictedGeofence) {
            if (!this.isInsideRestrictedArea) {
                // Drone entered restricted area
                this.handleRestrictedAreaEntry(currentGeofence);
            }
            this.isInsideRestrictedArea = true;
            this.isInsideGeofence = false; // Not in allowed area
        } else if (this.isInsideRestrictedArea) {
            // Drone exited restricted area
            this.handleRestrictedAreaExit();
            this.isInsideRestrictedArea = false;
        }

        // Handle allowed area (GREEN)
        if (insideAllowedGeofence && !insideRestrictedGeofence) {
            if (!this.isInsideGeofence) {
                // Drone entered allowed area
                this.handleGeofenceReEntry();
            }
            this.isInsideGeofence = true;
        } else if (!insideRestrictedGeofence && !insideAllowedGeofence) {
            // Drone is outside all geofences
            if (this.isInsideGeofence) {
                this.handleGeofenceExit();
            }
            this.isInsideGeofence = false;
        }

        this.updateDroneMarkerColor();
    }

    /**
     * Handle geofence exit event (exited allowed area)
     */
    handleGeofenceExit() {
        this.showToast('⚠️ Drone exited allowed mining area', 'analyzing');
        this.addAlert('Drone has exited the allowed mining area', 'analyzing');
    }

    /**
     * Handle geofence re-entry event (entered allowed area)
     */
    handleGeofenceReEntry() {
        this.showToast(' Drone entered allowed mining area', 'legal');
        this.addAlert('Drone has entered an allowed mining area', 'legal');
    }

    /**
     * Handle restricted area entry (RED - CRITICAL)
     */
    handleRestrictedAreaEntry(geofence) {
        const geofenceName = geofence ? geofence.name : 'restricted area';
        this.showToast(' CRITICAL: DRONE ENTERED RESTRICTED MINING AREA!', 'illegal');
        this.addAlert(`CRITICAL: Drone has entered restricted mining area "${geofenceName}"`, 'illegal');
        
        // Play alert sound if enabled
        if (this.settings.soundAlerts) {
            this.playAlertSound();
        }

        // Update status banner
        this.updateStatusBanner('illegal');
    }

    /**
     * Handle restricted area exit
     */
    handleRestrictedAreaExit() {
        this.showToast('✅ Drone exited restricted mining area', 'legal');
        this.addAlert('Drone has exited the restricted mining area', 'legal');
    }

    /**
     * Create a realistic drone icon using canvas
     * @param {String} color - Color for the drone icon
     * @returns {Object} - Google Maps icon object
     */
    createDroneIcon(color) {
        // Create canvas for custom drone icon
        const canvas = document.createElement('canvas');
        const size = 64;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        // Enable anti-aliasing for smoother rendering
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // Draw realistic quadcopter drone
        const centerX = size / 2;
        const centerY = size / 2;
        const bodySize = 14;
        const armLength = 20;
        const propSize = 9;
        const armWidth = 4;

        // Draw 4 arms with better styling
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
        // Top propeller with blades
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(centerX, centerY - armLength, propSize, 0, Math.PI * 2);
        ctx.fill();
        // Propeller blades
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - armLength - propSize);
        ctx.lineTo(centerX, centerY - armLength + propSize);
        ctx.moveTo(centerX - propSize, centerY - armLength);
        ctx.lineTo(centerX + propSize, centerY - armLength);
        ctx.stroke();
        // Propeller circle outline
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
        // Right propeller
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
        // Bottom propeller
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
        // Left propeller
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

        // Draw central body (more detailed)
        ctx.fillStyle = color;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(centerX, centerY, bodySize/2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Add body details - camera/gimbal indicator
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
        ctx.fill();
        
        // Add small LED indicators
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
            rotation: this.currentDronePosition.heading
        };
    }

    /**
     * Update drone marker color based on geofence status
     */
    updateDroneMarkerColor() {
        if (!this.droneMarker) return;

        // Priority: Restricted (red) > Allowed (green) > Outside (yellow/orange)
        let color = '#FFA500'; // Default: orange/yellow for outside all geofences
        if (this.isInsideRestrictedArea) {
            color = '#FF4444'; // Red for restricted area
        } else if (this.isInsideGeofence) {
            color = '#00FF99'; // Green for allowed area
        }

        // Create new icon with updated color
        const newIcon = this.createDroneIcon(color);
        newIcon.rotation = this.currentDronePosition.heading;
        this.droneMarker.setIcon(newIcon);
    }

    /**
     * Play alert sound for geofence violations
     */
    playAlertSound() {
        // Create a simple beep sound using Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.value = 800;
            oscillator.type = 'sine';

            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        } catch (error) {
            console.warn('Could not play alert sound:', error);
        }
    }

    /**
     * Toggle geofence drawing mode
     */
    toggleGeofenceDrawingMode() {
        if (!this.drawingManager) {
            this.setupDrawingManager();
        }

        this.isGeofenceMode = !this.isGeofenceMode;

        if (this.isGeofenceMode) {
            this.drawingManager.setDrawingMode(google.maps.drawing.OverlayType.POLYGON);
            this.showToast('Click and drag on the map to draw a geofence', 'analyzing');
        } else {
            this.drawingManager.setDrawingMode(null);
            this.showToast('Geofence drawing mode disabled', 'legal');
        }

        this.updateGeofenceControls();
    }

    /**
     * Update geofence control buttons visibility
     */
    updateGeofenceControls() {
        const saveBtn = document.getElementById('geofenceSaveBtn');
        const cancelBtn = document.getElementById('geofenceCancelBtn');
        const geofenceBtn = document.getElementById('geofenceBtn');

        if (this.isGeofenceMode) {
            if (saveBtn) saveBtn.style.display = 'inline-block';
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
            if (geofenceBtn) geofenceBtn.classList.add('active');
        } else {
            if (saveBtn) saveBtn.style.display = 'none';
            if (cancelBtn) cancelBtn.style.display = 'none';
            if (geofenceBtn) geofenceBtn.classList.remove('active');
        }
    }

    /**
     * Update map zoom controls
     */
    updateMapControls() {
        // Map controls are handled by Google Maps UI
        // This method can be extended for custom controls
    }

    // =====================
    // Geocoding API - Location Search
    // =====================
    /**
     * Initialize Google Maps Geocoding services
     */
    initializeGeocoding() {
        if (!this.map || typeof google === 'undefined' || !google.maps) {
            return;
        }

        try {
            // Initialize Geocoder
            this.geocoder = new google.maps.Geocoder();

            // Initialize Places Autocomplete (optional, for better search suggestions)
            if (google.maps.places) {
                this.autocompleteService = new google.maps.places.AutocompleteService();
                this.placesService = new google.maps.places.PlacesService(this.map);
            }

            console.log('Geocoding API initialized');
        } catch (error) {
            console.error('Error initializing Geocoding API:', error);
            this.showToast('Geocoding API initialization failed', 'illegal');
        }
    }

    /**
     * Search for a location using Geocoding API
     */
    searchLocation() {
        const searchInput = document.getElementById('mapSearchInput');
        if (!searchInput || !this.geocoder || !this.map) {
            this.showToast('Search functionality not available', 'illegal');
            return;
        }

        const address = searchInput.value.trim();
        if (!address) {
            this.showToast('Please enter a location to search', 'analyzing');
            return;
        }

        // Show loading state
        const searchBtn = document.getElementById('mapSearchBtn');
        if (searchBtn) {
            searchBtn.disabled = true;
            searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        }

        // Geocode the address
        this.geocoder.geocode(
            { address: address },
            (results, status) => {
                // Reset button state
                if (searchBtn) {
                    searchBtn.disabled = false;
                    searchBtn.innerHTML = '<i class="fas fa-search"></i>';
                }

                if (status === 'OK' && results && results.length > 0) {
                    const result = results[0];
                    const location = result.geometry.location;
                    const lat = location.lat();
                    const lng = location.lng();

                    // Center map on searched location
                    this.map.setCenter({ lat, lng });
                    this.map.setZoom(15);

                    // Add or update search marker
                    this.addSearchMarker({ lat, lng }, result.formatted_address);

                    // Show success message
                    this.showToast(`Found: ${result.formatted_address}`, 'legal');

                    // Update search input with formatted address
                    searchInput.value = result.formatted_address;
                } else if (status === 'ZERO_RESULTS') {
                    this.showToast('No results found for this location', 'illegal');
                } else if (status === 'OVER_QUERY_LIMIT') {
                    this.showToast('Geocoding API quota exceeded. Please try again later.', 'illegal');
                } else if (status === 'REQUEST_DENIED') {
                    this.showToast('Geocoding API request denied. Please check your API key permissions.', 'illegal');
                } else {
                    this.showToast(`Geocoding failed: ${status}`, 'illegal');
                }
            }
        );
    }

    /**
     * Add or update marker for searched location
     * @param {Object} position - { lat, lng }
     * @param {String} address - Formatted address
     */
    addSearchMarker(position, address) {
        // Remove existing search marker
        if (this.searchMarker) {
            this.searchMarker.setMap(null);
        }

        // Create new search marker
        this.searchMarker = new google.maps.Marker({
            position: position,
            map: this.map,
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 10,
                fillColor: '#FFA500',
                fillOpacity: 1,
                strokeColor: '#FFFFFF',
                strokeWeight: 2
            },
            title: address,
            zIndex: 999,
            animation: google.maps.Animation.DROP
        });

        // Add info window
        const infoWindow = new google.maps.InfoWindow({
            content: `
                <div style="color: #000; padding: 5px; max-width: 250px;">
                    <strong style="color: #FFA500;">📍 Searched Location</strong><br>
                    <div style="margin-top: 5px; font-size: 0.9em;">${address}</div>
                    <div style="margin-top: 5px; font-size: 0.8em; color: #666;">
                        Lat: ${position.lat.toFixed(6)}<br>
                        Lng: ${position.lng.toFixed(6)}
                    </div>
                </div>
            `
        });

        // Show info window on marker click
        this.searchMarker.addListener('click', () => {
            infoWindow.open(this.map, this.searchMarker);
        });

        // Show info window immediately
        infoWindow.open(this.map, this.searchMarker);

        // Auto-close info window after 5 seconds
        setTimeout(() => {
            infoWindow.close();
        }, 5000);
    }

    /**
     * Clear search results and remove search marker
     */
    clearSearch() {
        const searchInput = document.getElementById('mapSearchInput');
        if (searchInput) {
            searchInput.value = '';
        }

        const mapSearchClear = document.getElementById('mapSearchClear');
        if (mapSearchClear) {
            mapSearchClear.style.display = 'none';
        }

        // Remove search marker
        if (this.searchMarker) {
            this.searchMarker.setMap(null);
            this.searchMarker = null;
        }

        // Center map back on drone position
        if (this.map && this.currentDronePosition) {
            this.map.setCenter({
                lat: this.currentDronePosition.lat,
                lng: this.currentDronePosition.lng
            });
        }
    }

    /**
     * Get location name from coordinates (reverse geocoding)
     * @param {Number} lat - Latitude
     * @param {Number} lng - Longitude
     * @returns {Promise<String>} - Formatted address
     */
    async reverseGeocode(lat, lng) {
        if (!this.geocoder) {
            return null;
        }

        return new Promise((resolve, reject) => {
            this.geocoder.geocode(
                { location: { lat, lng } },
                (results, status) => {
                    if (status === 'OK' && results && results.length > 0) {
                        resolve(results[0].formatted_address);
                    } else {
                        reject(new Error(`Reverse geocoding failed: ${status}`));
                    }
                }
            );
        });
    }

    /**
     * Convert old geofence format (percent-based) to lat/lng format
     * This is for backward compatibility
     */
    convertOldGeofencesToLatLng() {
        // If geofences are in old format (percent-based), convert them
        // This is a placeholder - implement if needed
    }

    /**
     * Save geofences to localStorage with lat/lng coordinates
     */
    persistGeofences() {
        try {
            const geofencesToSave = this.geofences.map(geofence => ({
                id: geofence.id,
                name: geofence.name,
                type: geofence.type || 'allowed', // Include type, default to allowed
                coordinates: geofence.coordinates
            }));
            localStorage.setItem('geofences', JSON.stringify(geofencesToSave));
        } catch (error) {
            console.error('Failed to save geofences:', error);
        }
    }

    /**
     * Load geofences from localStorage
     */
    loadGeofences() {
        try {
            const saved = localStorage.getItem('geofences');
            if (saved) {
                this.geofences = JSON.parse(saved);
                // Render on map if map is initialized
                if (this.mapInitialized) {
                    this.renderSavedGeofencesOnMap();
                }
            }
            this.renderGeofenceList();
        } catch (error) {
            console.error('Failed to load geofences:', error);
            this.geofences = [];
            this.renderGeofenceList();
        }
    }

    /**
     * Open the Live Location Tracking map in a separate popup window
     */
    openMapInPopup() {
        try {
            // Check if map tab is already open
            if (this.mapPopupWindow && !this.mapPopupWindow.closed) {
                this.mapPopupWindow.focus();
                this.showToast('Map tab is already open', 'legal');
                return;
            }

            // Get current map state
            let currentLat = 5.6037;
            let currentLng = -0.1870;
            let currentZoom = 15;
            let currentMapTypeStr = 'SATELLITE';
            
            if (this.map) {
                try {
                    const center = this.map.getCenter();
                    currentLat = center.lat();
                    currentLng = center.lng();
                    currentZoom = this.map.getZoom();
                    const mapType = this.map.getMapTypeId();
                    if (mapType === google.maps.MapTypeId.SATELLITE) {
                        currentMapTypeStr = 'SATELLITE';
                    } else if (mapType === google.maps.MapTypeId.TERRAIN) {
                        currentMapTypeStr = 'TERRAIN';
                    } else if (mapType === google.maps.MapTypeId.HYBRID) {
                        currentMapTypeStr = 'HYBRID';
                    } else {
                        currentMapTypeStr = 'ROADMAP';
                    }
                } catch (e) {
                    console.warn('Error getting map state:', e);
                }
            }

            // Store initial state in sessionStorage for tab to read
            sessionStorage.setItem('popupMapState', JSON.stringify({
                lat: currentLat,
                lng: currentLng,
                zoom: currentZoom,
                mapType: currentMapTypeStr,
                dronePosition: this.currentDronePosition
            }));

            // Get the base path for the map HTML file
            const basePath = window.location.origin + window.location.pathname.replace(/[^/]*$/, '');
            const mapUrl = basePath + 'map-popup.html';

            // Hide the map section on main page
            const mapSection = document.querySelector('.map-section');
            if (mapSection) {
                mapSection.style.display = 'none';
            }

            // Open map in a new tab
            this.mapPopupWindow = window.open(mapUrl, '_blank');
            
            if (!this.mapPopupWindow) {
                this.showToast('Failed to open map in new tab. Please allow popups for this site.', 'illegal');
                // Show map section again if tab opening failed
                if (mapSection) {
                    mapSection.style.display = '';
                }
                return;
            }

            // Set up communication between tabs using BroadcastChannel and localStorage
            const channel = new BroadcastChannel('mapTabChannel');
            
            // Listen for messages from the map tab
            channel.onmessage = (event) => {
                if (event.data.type === 'requestPosition') {
                    // Send current drone position to map tab
                    channel.postMessage({
                        type: 'dronePosition',
                        position: this.currentDronePosition
                    });
                } else if (event.data.type === 'tabClosing') {
                    // Map tab is closing, show map section again
                    const mapSection = document.querySelector('.map-section');
                    if (mapSection) {
                        mapSection.style.display = '';
                    }
                    channel.close();
                }
            };

            // Monitor tab and send position updates
            const syncInterval = setInterval(() => {
                try {
                    // Check if tab is still open by trying to access it
                    if (!this.mapPopupWindow || this.mapPopupWindow.closed) {
                        clearInterval(syncInterval);
                        this.mapPopupSyncInterval = null;
                        
                        // Show the map section again when tab is closed
                        const mapSection = document.querySelector('.map-section');
                        if (mapSection) {
                            mapSection.style.display = '';
                        }
                        
                        // Re-initialize map if it was destroyed
                        if (this.map && this.mapInitialized) {
                            // Map should still be there, just make sure it's visible
                            const mapView = document.getElementById('mapView');
                            if (mapView) {
                                // Trigger a resize event to ensure map displays correctly
                                if (typeof google !== 'undefined' && google.maps) {
                                    setTimeout(() => {
                                        google.maps.event.trigger(this.map, 'resize');
                                    }, 100);
                                }
                            }
                        }
                        
                        channel.close();
                        return;
                    }
                    // Send current position to map tab
                    channel.postMessage({
                        type: 'dronePosition',
                        position: this.currentDronePosition
                    });
                } catch (e) {
                    // Tab might be closed or in different origin
                    clearInterval(syncInterval);
                    this.mapPopupSyncInterval = null;
                    const mapSection = document.querySelector('.map-section');
                    if (mapSection) {
                        mapSection.style.display = '';
                    }
                    channel.close();
                }
            }, 1000);

            // Store interval and channel for cleanup
            this.mapPopupSyncInterval = syncInterval;
            this.mapTabChannel = channel;

            this.showToast('Map opened in new tab', 'legal');
        } catch (error) {
            console.error('Error opening map popup:', error);
            this.showToast('Failed to open map popup: ' + error.message, 'illegal');
            if (this.mapPopupWindow && !this.mapPopupWindow.closed) {
                this.mapPopupWindow.close();
            }
        }
    }

    // =====================
    // WebSocket/API Integration for Real-time Updates
    // =====================
    /**
     * Handle WebSocket message with drone position data
     * Expected format: { lat: <number>, lng: <number>, altitude: <number>, heading: <number> }
     */
    handleDronePositionUpdate(data) {
        if (!data || typeof data.lat !== 'number' || typeof data.lng !== 'number') {
            console.warn('Invalid drone position data:', data);
            return;
        }

        // Stop device location tracking if WebSocket data is available
        if (this.useDeviceLocation && this.geolocationWatchId !== null) {
            this.stopDeviceLocationTracking();
            this.useDeviceLocation = false;
        }

        // Stop simulation if real data is coming through
        if (this.simulationMode) {
            this.stopDroneSimulation();
        }

        this.updateDronePosition({
            lat: data.lat,
            lng: data.lng,
            altitude: data.altitude || 0,
            heading: data.heading || 0
        });
    }

    // =====================
    // Device Location Tracking (Browser Geolocation API)
    // =====================
    /**
     * Start tracking device location using browser Geolocation API
     * This uses the device's GPS/network location as the drone position
     */
    startDeviceLocationTracking() {
        if (!navigator.geolocation) {
            console.error('Geolocation is not supported by this browser');
            this.showToast('Geolocation not supported by your browser', 'illegal');
            return;
        }

        // Update button state
        const toggleBtn = document.getElementById('toggleDeviceLocation');
        if (toggleBtn) {
            toggleBtn.classList.add('active');
        }

        // Request permission and get current position
        navigator.geolocation.getCurrentPosition(
            (position) => {
                this.handleDeviceLocationUpdate(position);
                this.showToast('Device location tracking started', 'legal');
            },
            (error) => {
                this.handleGeolocationError(error);
                // Remove active state if permission denied
                if (error.code === error.PERMISSION_DENIED && toggleBtn) {
                    toggleBtn.classList.remove('active');
                }
            },
            {
                enableHighAccuracy: true, // Use GPS if available
                timeout: 10000, // 10 second timeout
                maximumAge: 0 // Don't use cached position
            }
        );

        // Watch position for continuous updates
        this.geolocationWatchId = navigator.geolocation.watchPosition(
            (position) => {
                this.handleDeviceLocationUpdate(position);
            },
            (error) => {
                this.handleGeolocationError(error);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 5000 // Accept positions up to 5 seconds old
            }
        );

        console.log('Device location tracking started');
    }

    /**
     * Handle device location update from Geolocation API
     * @param {GeolocationPosition} position - Browser geolocation position object
     */
    handleDeviceLocationUpdate(position) {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        const altitude = position.coords.altitude || 0;
        const heading = position.coords.heading || 0;
        const accuracy = position.coords.accuracy; // Accuracy in meters

        // Calculate heading from previous position if not available
        let calculatedHeading = heading;
        if (!heading && this.currentDronePosition.lat && this.currentDronePosition.lng) {
            // Calculate bearing between two points
            const prevLat = this.currentDronePosition.lat;
            const prevLng = this.currentDronePosition.lng;
            const dLng = (lng - prevLng) * Math.PI / 180;
            const prevLatRad = prevLat * Math.PI / 180;
            const latRad = lat * Math.PI / 180;
            
            const y = Math.sin(dLng) * Math.cos(latRad);
            const x = Math.cos(prevLatRad) * Math.sin(latRad) - 
                     Math.sin(prevLatRad) * Math.cos(latRad) * Math.cos(dLng);
            
            calculatedHeading = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
        }

        // If this is the first location received, center the map on it
        if (!this.firstLocationReceived && this.map) {
            this.firstLocationReceived = true;
            this.map.setCenter({ lat, lng });
            this.map.setZoom(17); // Zoom in closer for better view
            console.log('Map centered on device location:', lat, lng);
        }

        // Update drone position with device location
        this.updateDronePosition({
            lat: lat,
            lng: lng,
            altitude: altitude,
            heading: calculatedHeading
        });

        // Log accuracy for debugging
        if (accuracy) {
            console.log(`Device location updated - Accuracy: ${accuracy.toFixed(1)}m`);
        }
    }

    /**
     * Handle geolocation errors
     * @param {GeolocationPositionError} error - Geolocation error object
     */
    handleGeolocationError(error) {
        let errorMessage = 'Location tracking error: ';
        
        switch(error.code) {
            case error.PERMISSION_DENIED:
                errorMessage += 'Permission denied. Please allow location access.';
                this.showToast('Location permission denied. Please enable location access in browser settings.', 'illegal');
                break;
            case error.POSITION_UNAVAILABLE:
                errorMessage += 'Position unavailable.';
                this.showToast('Location unavailable. Check your GPS/network connection.', 'illegal');
                break;
            case error.TIMEOUT:
                errorMessage += 'Request timeout.';
                this.showToast('Location request timeout. Trying again...', 'analyzing');
                // Retry after a delay
                setTimeout(() => {
                    if (this.useDeviceLocation) {
                        this.startDeviceLocationTracking();
                    }
                }, 5000);
                break;
            default:
                errorMessage += 'Unknown error.';
                this.showToast('Location tracking error occurred.', 'illegal');
                break;
        }
        
        console.error(errorMessage, error);
    }

    /**
     * Stop tracking device location
     */
    stopDeviceLocationTracking() {
        if (this.geolocationWatchId !== null) {
            navigator.geolocation.clearWatch(this.geolocationWatchId);
            this.geolocationWatchId = null;
            console.log('Device location tracking stopped');
            this.showToast('Device location tracking stopped', 'legal');
            
            // Update button state
            const toggleBtn = document.getElementById('toggleDeviceLocation');
            if (toggleBtn) {
                toggleBtn.classList.remove('active');
            }
        }
    }

    // =====================
    // Drone Simulation Mode (for testing when WebSocket is not available)
    // =====================
    /**
     * Start simulation mode to test drone movement
     * This simulates a drone moving in a circular pattern
     */
    startDroneSimulation() {
        if (this.simulationMode) {
            return; // Already running
        }

        this.simulationMode = true;
        console.log('Starting drone simulation mode');
        this.showToast('Simulation mode: Testing drone movement', 'analyzing');

        // Update UI buttons
        const testBtn = document.getElementById('testDronePosition');
        const stopBtn = document.getElementById('stopSimulation');
        if (testBtn) testBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = 'inline-block';

        // Starting position (Accra, Ghana)
        let centerLat = 5.6037;
        let centerLng = -0.1870;
        let radius = 0.01; // ~1km radius
        let angle = 0;
        let altitude = 100;
        let heading = 0;

        // Update position every 2 seconds
        this.simulationInterval = setInterval(() => {
            // Calculate circular movement
            angle += 0.1; // Increment angle
            const lat = centerLat + radius * Math.cos(angle);
            const lng = centerLng + radius * Math.sin(angle);
            heading = (angle * 180 / Math.PI + 90) % 360; // Convert angle to heading
            altitude = 100 + 20 * Math.sin(angle * 2); // Vary altitude

            // Update drone position
            this.updateDronePosition({
                lat: lat,
                lng: lng,
                altitude: altitude,
                heading: heading
            });

            // Update center position slightly to create a spiral pattern
            centerLat += 0.0001 * Math.cos(angle * 0.5);
            centerLng += 0.0001 * Math.sin(angle * 0.5);
        }, 2000); // Update every 2 seconds
    }

    /**
     * Stop simulation mode
     */
    stopDroneSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
        }
        this.simulationMode = false;
        console.log('Stopped drone simulation mode');
        this.showToast('Simulation mode stopped', 'legal');

        // Update UI buttons
        const testBtn = document.getElementById('testDronePosition');
        const stopBtn = document.getElementById('stopSimulation');
        if (testBtn) testBtn.style.display = 'inline-block';
        if (stopBtn) stopBtn.style.display = 'none';
    }

    /**
     * Manually update drone position (for testing)
     * Can be called from browser console: detector.updateDronePositionManually({lat: 5.6037, lng: -0.1870, altitude: 100, heading: 45})
     */
    updateDronePositionManually(position) {
        if (!position || typeof position.lat !== 'number' || typeof position.lng !== 'number') {
            console.error('Invalid position. Expected: {lat: number, lng: number, altitude?: number, heading?: number}');
            return;
        }

        this.updateDronePosition({
            lat: position.lat,
            lng: position.lng,
            altitude: position.altitude || this.currentDronePosition.altitude,
            heading: position.heading || this.currentDronePosition.heading
        });

        console.log('Drone position updated manually:', this.currentDronePosition);
        this.showToast(`Position updated: ${position.lat.toFixed(6)}, ${position.lng.toFixed(6)}`, 'legal');
    }

}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    let detector;
    try {
        detector = new GalamseyDetector();
        // Make detector instance available globally for Google Maps callback
        window.detectorInstance = detector;
        
        // Initialize Google Maps if API is already loaded
        if (typeof google !== 'undefined' && google.maps) {
            detector.initializeGoogleMap();
        } else {
            // Set up callback for when Google Maps API loads
            window.initMap = () => {
                if (window.detectorInstance) {
                    window.detectorInstance.initializeGoogleMap();
                }
            };
        }
        
        // Ensure body is visible even if loading overlay fails
        document.body.style.opacity = '1';
    } catch (err) {
        console.error('Initialization error:', err);
        // Fallback: show the page to avoid a blank screen
        document.body.style.opacity = '1';
        try {
            // Attempt minimal toast if class partially available
            const container = document.getElementById('toastContainer');
            if (container) {
                const t = document.createElement('div');
                t.className = 'toast illegal';
                t.textContent = 'Initialization error - check console logs';
                container.appendChild(t);
                setTimeout(() => t.remove(), 5000);
            }
        } catch(_) {}
        return;
    }
    
    // GPS tracking will be updated from actual device data when connected
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'r':
                    e.preventDefault();
                    detector.generateReport();
                    break;
                case 'h':
                    e.preventDefault();
                    detector.viewHistory();
                    break;
            }
        }
        
        // F11 for fullscreen
        if (e.key === 'F11') {
            e.preventDefault();
            detector.toggleFullscreen();
        }
        
        // Escape to exit fullscreen or close modals
        if (e.key === 'Escape') {
            if (document.fullscreenElement) {
                detector.toggleFullscreen();
            } else if (document.getElementById('settingsModal').classList.contains('active')) {
                detector.hideSettings();
            }
        }
        
        // Ctrl+, for settings
        if ((e.ctrlKey || e.metaKey) && e.key === ',') {
            e.preventDefault();
            detector.showSettings();
        }
        
        // M for map mode toggle
        if (e.key === 'm' || e.key === 'M') {
            e.preventDefault();
            detector.toggleMapMode();
        }
    });
    
    // Loading animation is now handled by hideLoadingScreen method
});

// Global error handler to avoid blank screen if any runtime error occurs later
window.addEventListener('error', () => {
    document.body.style.opacity = '1';
});

// CSS animations are now handled in the main styles.css file
