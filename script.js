/**
 * AI FACE-MESH SCULPTOR - CLIENT ENGINE (DECOUPLED & FAIL-SAFE VERSION)
 * Role: Premium Creative Frontend Developer, WebGL Expert, & Machine Learning Engineer
 * 
 * Features:
 * 1. Decoupled load: Three.js and animations boot instantly. The loader screen fades out immediately.
 * 2. Asynchronous background load: Webcam connection and TensorFlow.js Face Mesh compile in the background.
 * 3. Telemetry tracking updates seamlessly (e.g. "COMPILING MODEL", "ACTIVE", "MOUSE MODE").
 * 4. Premium Mouse Fallback: If no camera or during loading, mouse coordinates drive Yaw/Pitch, and click-holds sculpt the mouth cavity!
 * 5. Dynamic Web Audio synthesizer with LFO lowpass filter resonance and FM modulation.
 */

// Global State
const state = {
  // Telemetry Metrics
  metrics: {
    mouthOpen: 0,
    smileWidth: 0,
    blinkOpenLeft: 1,
    blinkOpenRight: 1,
    isBlinking: false,
    yaw: 0,   // Horizontal head rotation
    pitch: 0, // Vertical head rotation
    roll: 0,  // Tilted head rotation
    faceDetected: false,
    lastMesh: null
  },
  // Smooth LERP Variables
  smoothed: {
    mouthOpen: 0,
    smileWidth: 0,
    blink: 0,
    yaw: 0,
    pitch: 0,
    roll: 0,
    faceMorph: 0
  },
  // Load States
  cameraActive: false,
  modelLoaded: false,
  // Performance
  fps: 0,
  lastFrameTime: performance.now(),
  frameCount: 0,
  fpsIntervalTime: performance.now()
};

// DOM Elements
const webcamEl = document.getElementById('webcam');
const webcamPreviewCanvas = document.getElementById('webcam-preview-canvas');
const webcamPreviewCtx = webcamPreviewCanvas.getContext('2d');
const webcamPreviewContainer = document.getElementById('webcam-preview-container');
const trackingStatus = document.getElementById('tracking-status');
const audioShield = document.getElementById('audio-shield');
const loadingScreen = document.getElementById('loading-screen');
const fpsCounter = document.getElementById('fps-counter');

// Loader DOM Elements & Progress Utilities
const progressFill = document.getElementById('progress-fill');
const progressPercent = document.getElementById('progress-percent');
const loadingSubtext = document.querySelector('.loading-subtext');

function updateLoadingProgress(percent, statusText) {
  if (progressFill) progressFill.style.width = percent + '%';
  if (progressPercent) progressPercent.textContent = percent + '%';
  if (loadingSubtext && statusText) loadingSubtext.textContent = statusText;
  debugLog(statusText || ('Progress: ' + percent + '%'));
}

function finalizeLoading(success, statusText) {
  updateLoadingProgress(100, statusText || "COMPILATION SUCCESSFUL // PREPARING NEURAL EXPERIENCE");
  debugLog(success ? '✅ LOAD COMPLETE — FACE TRACKING ACTIVE' : '⚠️ LOAD COMPLETE — MOUSE MODE');
  setTimeout(() => {
    loadingScreen.classList.add('fade-out');
    // Show audio shield prompting synthesizer activation
    audioShield.classList.remove('hidden');
  }, 1000);
}

// ==========================================================================
// DIAGNOSTIC LOGGING
// ==========================================================================
function debugLog(msg) {
  console.log('[SCULPTOR]', msg);
}

// 3D Scene Variables
let scene, camera, renderer, sculpture, geometry, material;
let originalPositions;
let pointLightMagenta, pointLightCyan, pointLightBlue;
let vertexToLandmarkMap = null;

// Audio variables
let audioCtx = null;
let masterGain = null;
let droneOsc = null;
let subOsc = null;
let vocalFilter = null;
let synthFilter = null;
let lfo = null;
let lfoGain = null;
let audioInitialized = false;

// Mouse Interaction States (Fallback)
let mouseX = 0;
let mouseY = 0;
let isMouseDown = false;

// Track Mouse Movement
window.addEventListener('mousemove', (e) => {
  mouseX = (e.clientX / window.innerWidth) * 2 - 1; // -1 to 1
  mouseY = -(e.clientY / window.innerHeight) * 2 + 1; // -1 to 1
});

window.addEventListener('mousedown', () => {
  isMouseDown = true;
});

window.addEventListener('mouseup', () => {
  isMouseDown = false;
});

// ==========================================================================
// 1. WEBCAM & HARDWARE HANDLER (ASYNC BACKGROUND)
// ==========================================================================
async function initWebcam() {
  try {
    debugLog('Checking camera API availability...');
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      debugLog('❌ Camera API unavailable (insecure context or unsupported browser)');
      state.cameraActive = false;
      return false;
    }

    debugLog('Requesting camera permission via getUserMedia...');
    
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      },
      audio: false
    });
    
    debugLog('✅ Camera stream acquired. Attaching to video element...');
    webcamEl.srcObject = stream;
    
    // CRITICAL FIX: Check readyState BEFORE attaching the event handler.
    // If the browser loads metadata instantly (common for local webcams),
    // the 'loadedmetadata' event fires BEFORE onloadedmetadata is assigned,
    // causing the Promise to hang forever and silently breaking everything.
    if (webcamEl.readyState < 1) {
      debugLog('Waiting for video metadata...');
      await new Promise((resolve) => {
        webcamEl.addEventListener('loadedmetadata', resolve, { once: true });
      });
    } else {
      debugLog('Video metadata already loaded (fast path).');
    }
    
    debugLog('Starting video playback...');
    await webcamEl.play();
    
    // Setup preview canvas proportions
    webcamPreviewCanvas.width = 140;
    webcamPreviewCanvas.height = 105;
    state.cameraActive = true;
    webcamPreviewContainer.style.display = 'block';
    
    debugLog('✅ Webcam LIVE — preview canvas ready (' + webcamEl.videoWidth + 'x' + webcamEl.videoHeight + ')');
    return true;
  } catch (err) {
    debugLog('❌ Camera error: ' + (err.name || '') + ' ' + (err.message || err));
    state.cameraActive = false;
    return false;
  }
}

// ==========================================================================
// 2. ML INFERENCE LOOP (TENSORFLOW / MEDIAPIPE FACE MESH)
// ==========================================================================
let faceMeshModel;

async function initFaceMesh() {
  let hasResolved = false; // Guard against double-resolve race condition

  return new Promise(async (resolve) => {
    // 60-second timeout — FaceMesh model download + WebGL shader compilation
    // routinely takes 15-30s on average connections. 3.5s was far too aggressive.
    const fallbackTimeout = setTimeout(() => {
      if (hasResolved) return;
      hasResolved = true;
      console.warn("FaceMesh model loading taking too long (60s). Falling back to Mouse Mode.");
      state.modelLoaded = true;
      finalizeLoading(false, "COMPILATION TIMED OUT // ACTIVATING CYBER MOUSE MODE");
      resolve(false);
    }, 60000);

    try {
      debugLog('Initializing TensorFlow.js backend...');
      updateLoadingProgress(60, "INITIALIZING TENSORFLOW ENGINE // COMPILING SHADERS");
      
      // Let TFJS automatically negotiate best available backend (WebGL -> CPU)
      await tf.ready();
      debugLog('✅ TF.js ready — backend: ' + tf.getBackend());
      updateLoadingProgress(75, "TENSORFLOW ENGINE ACTIVE // DOWNLOADING NEURAL MESH WEIGHTS");

      // 2. Load face landmarks detection package (MediaPipe FaceMesh)
      faceMeshModel = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.MEDIAPIPE_FACEMESH
      );
      
      clearTimeout(fallbackTimeout);
      if (hasResolved) return; // Timeout already fired, don't double-resolve
      hasResolved = true;

      state.modelLoaded = true;
      debugLog('✅ FaceMesh model compiled! Starting face tracking loop...');
      updateLoadingProgress(90, "FACE TRACKER READY // COUPLING SOUNDSCAPE WAVE OSCILLATORS");
      
      // Start continuous tracking loop
      trackFace();
      
      setTimeout(() => {
        finalizeLoading(true, "COMPILATION SUCCESSFUL // PREPARING NEURAL EXPERIENCE");
        resolve(true);
      }, 500);
    } catch (err) {
      clearTimeout(fallbackTimeout);
      if (hasResolved) return;
      hasResolved = true;
      console.warn("Failed to compile FaceMesh model. Transitioning to Mouse Fallback mode.", err);
      state.modelLoaded = true;
      finalizeLoading(false, "MODEL COMPILATION FAILED // ACTIVATING CYBER MOUSE MODE");
      resolve(false);
    }
  });
}

async function trackFace() {
  if (state.cameraActive && !webcamEl.paused && !webcamEl.ended) {
    try {
      const predictions = await faceMeshModel.estimateFaces({
        input: webcamEl,
        returnTensors: false,
        flipHorizontal: false
      });

      if (predictions && predictions.length > 0) {
        state.metrics.faceDetected = true;
        state.metrics.lastMesh = predictions[0].scaledMesh;
        processFaceLandmarks(predictions[0].scaledMesh);
      } else {
        state.metrics.faceDetected = false;
        state.metrics.lastMesh = null;
      }
    } catch (err) {
      console.warn("Frame estimation issue:", err);
    }
  }
  requestAnimationFrame(trackFace);
}

/**
 * Map raw 3D vectors to facial expressions & head rotation
 */
function processFaceLandmarks(landmarks) {
  // Helper: Euclidean distance in 3D
  const dist = (p1, p2) => {
    return Math.sqrt(
      (p1[0] - p2[0])**2 +
      (p1[1] - p2[1])**2 +
      (p1[2] - p2[2])**2
    );
  };

  // 1. Establish stable face scale (Left outer eye corner 33 to Right outer eye corner 263)
  const faceScale = dist(landmarks[33], landmarks[263]);

  // 2. Mouth Openness: distance between upper/lower lip centers (Landmarks 13 & 14)
  const mouthGap = dist(landmarks[13], landmarks[14]);
  const normMouthOpen = Math.min(Math.max((mouthGap / faceScale - 0.05) / 0.40, 0), 1);
  state.metrics.mouthOpen = normMouthOpen;

  // 3. Smile Width: distance between mouth outer corners (Landmarks 61 & 291)
  const smileDist = dist(landmarks[61], landmarks[291]);
  const normSmileWidth = Math.min(Math.max((smileDist / faceScale - 0.48) / 0.17, 0), 1);
  state.metrics.smileWidth = normSmileWidth;

  // 4. Eye Openness (Blink):
  // Left eyelid: 159 (top), 145 (bottom)
  const leftEyeOpen = dist(landmarks[159], landmarks[145]) / faceScale;
  const rightEyeOpen = dist(landmarks[386], landmarks[374]) / faceScale;
  
  const normEyeOpenLeft = Math.min(Math.max((leftEyeOpen - 0.018) / 0.065, 0), 1);
  const normEyeOpenRight = Math.min(Math.max((rightEyeOpen - 0.018) / 0.065, 0), 1);
  
  state.metrics.blinkOpenLeft = normEyeOpenLeft;
  state.metrics.blinkOpenRight = normEyeOpenRight;
  
  // Detect full blink (either eye closed below 0.20 threshold)
  state.metrics.isBlinking = (normEyeOpenLeft < 0.20 || normEyeOpenRight < 0.20);

  // 5. Head Rotation (Yaw / Pitch / Roll):
  const nose = landmarks[4];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  
  // Yaw (Left/Right)
  const noseLeftDist = dist(nose, leftCheek);
  const noseRightDist = dist(nose, rightCheek);
  const yawRatio = noseLeftDist / (noseLeftDist + noseRightDist);
  state.metrics.yaw = (yawRatio - 0.5) * 1.5;

  // Pitch (Up/Down)
  const forehead = landmarks[10];
  const chin = landmarks[152];
  const noseTopDist = dist(nose, forehead);
  const noseBottomDist = dist(nose, chin);
  const pitchRatio = noseTopDist / (noseTopDist + noseBottomDist);
  state.metrics.pitch = (pitchRatio - 0.37) * 2.5;

  // Roll (Tilt)
  const rollDY = landmarks[263][1] - landmarks[33][1];
  const rollDX = landmarks[263][0] - landmarks[33][0];
  state.metrics.roll = Math.atan2(rollDY, rollDX);
}

/**
 * Draws the user's face to the mirrored preview canvas in real-time.
 * Overlays key glowing neural landmarks when face tracking is active.
 */
function updateWebcamPreview() {
  if (!state.cameraActive) return;

  const w = webcamPreviewCanvas.width;
  const h = webcamPreviewCanvas.height;

  // 1. Draw mirrored raw video frame
  webcamPreviewCtx.save();
  webcamPreviewCtx.translate(w, 0);
  webcamPreviewCtx.scale(-1, 1);
  webcamPreviewCtx.drawImage(webcamEl, 0, 0, w, h);
  webcamPreviewCtx.restore();

  // 2. Overlay neural mesh if active and mesh data exists
  if (state.metrics.faceDetected && state.metrics.lastMesh) {
    webcamPreviewContainer.classList.add('active');
    
    if (state.metrics.isBlinking) {
      webcamPreviewContainer.classList.add('blink-active');
    } else {
      webcamPreviewContainer.classList.remove('blink-active');
    }

    webcamPreviewCtx.fillStyle = state.metrics.isBlinking ? 'rgba(255, 0, 127, 0.85)' : 'rgba(0, 255, 204, 0.85)';
    webcamPreviewCtx.strokeStyle = 'rgba(0, 255, 204, 0.2)';
    webcamPreviewCtx.lineWidth = 0.5;

    const scaleX = w / 640;
    const scaleY = h / 480;

    const keyLandmarks = [
      4, 10, 152,
      33, 133, 145, 159, 263, 362, 374, 386,
      61, 291, 13, 14,
      234, 454
    ];

    keyLandmarks.forEach((idx) => {
      const pt = state.metrics.lastMesh[idx];
      if (pt) {
        const drawX = w - (pt[0] * scaleX);
        const drawY = pt[1] * scaleY;
        webcamPreviewCtx.beginPath();
        webcamPreviewCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
        webcamPreviewCtx.fill();
      }
    });

    const drawLine = (i1, i2) => {
      const pt1 = state.metrics.lastMesh[i1];
      const pt2 = state.metrics.lastMesh[i2];
      if (pt1 && pt2) {
        webcamPreviewCtx.beginPath();
        webcamPreviewCtx.moveTo(w - (pt1[0] * scaleX), pt1[1] * scaleY);
        webcamPreviewCtx.lineTo(w - (pt2[0] * scaleX), pt2[1] * scaleY);
        webcamPreviewCtx.stroke();
      }
    };

    drawLine(10, 4); drawLine(4, 152); drawLine(234, 4); drawLine(4, 454);
    drawLine(33, 159); drawLine(159, 133); drawLine(133, 145); drawLine(145, 33);
    drawLine(263, 386); drawLine(386, 362); drawLine(362, 374); drawLine(374, 263);
    drawLine(61, 13); drawLine(13, 291); drawLine(291, 14); drawLine(14, 61);
  } else {
    webcamPreviewContainer.classList.remove('active');
    webcamPreviewContainer.classList.remove('blink-active');
  }
}

// ==========================================================================
// 3. WEBGL 3D SCULPTURE ENGINE (THREE.JS)
// ==========================================================================
let faceMeshGeo, faceMeshMat, faceMeshObject;
let faceVertexPositions; // Float32Array for live 468*3 positions
let ambientLight, keyLight, backLight;

function initThreeEngine() {
  const canvas = document.getElementById('webgl-canvas');

  // 1. Scene & Fog Setup
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x010103, 0.08);

  // 2. Camera Setup
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 6.5);

  // 3. Renderer with high-end ACES Filmic tone mapping
  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance"
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;

  // 4. Idle Sphere (Icosahedron Detail 5) — shown when no face detected
  geometry = new THREE.IcosahedronGeometry(2.0, 5);
  originalPositions = geometry.attributes.position.array.slice();

  material = new THREE.MeshPhysicalMaterial({
    color: 0x00ffcc,
    roughness: 0.12,
    metalness: 0.88,
    clearcoat: 1.0,
    clearcoatRoughness: 0.08,
    transmission: 0.35,
    thickness: 2.0,
    ior: 1.5,
    flatShading: false,
    wireframe: false
  });

  sculpture = new THREE.Mesh(geometry, material);
  scene.add(sculpture);

  // 5. DIRECT FACE MESH — uses actual MediaPipe tessellation (898 triangles, 468 vertices)
  faceMeshGeo = new THREE.BufferGeometry();
  faceVertexPositions = new Float32Array(468 * 3); // Will be filled with live data
  faceUvs = new Float32Array(468 * 2); // Dynamic UVs mapping to video frame

  faceMeshGeo.setAttribute('position', new THREE.BufferAttribute(faceVertexPositions, 3));
  faceMeshGeo.setAttribute('uv', new THREE.BufferAttribute(faceUvs, 2));
  faceMeshGeo.setIndex(new THREE.BufferAttribute(FACE_MESH_TRIANGLES, 1));

  // Initialize Video Texture mapped to the hidden webcam video element
  const videoTexture = new THREE.VideoTexture(document.getElementById('webcam'));
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;
  videoTexture.format = THREE.RGBFormat;

  // Initialize material in Holographic Scanning Mode
  faceMeshMat = new THREE.MeshPhysicalMaterial({
    color: 0x00ffcc, // Cyan wireframe
    map: null, // No texture during scan
    roughness: 0.2,
    metalness: 0.8,
    clearcoat: 1.0,
    clearcoatRoughness: 0.1,
    transmission: 0.1,
    thickness: 1.0,
    ior: 1.5,
    wireframe: true, // Wireframe holographic effect
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.0 // Fades in when face detected
  });

  faceMeshObject = new THREE.Mesh(faceMeshGeo, faceMeshMat);
  scene.add(faceMeshObject);

  // 6. Cyberpunk Lighting Rig
  ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(2, 4, 5);
  scene.add(keyLight);

  // Back-fill light for face depth
  backLight = new THREE.DirectionalLight(0x6600ff, 0.5);
  backLight.position.set(-2, -1, -3);
  scene.add(backLight);

  pointLightMagenta = new THREE.PointLight(0xff007f, 4, 12);
  pointLightMagenta.position.set(-3, 2, 3);
  scene.add(pointLightMagenta);

  pointLightCyan = new THREE.PointLight(0x00ffcc, 4, 12);
  pointLightCyan.position.set(3, -2, 3);
  scene.add(pointLightCyan);

  pointLightBlue = new THREE.PointLight(0x1a00ff, 5, 15);
  pointLightBlue.position.set(0, 3, -4);
  scene.add(pointLightBlue);

  // 7. Orbit Controls for 3D Viewing
  orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.05;
  orbitControls.enabled = false; // Disabled during scan

  window.addEventListener('resize', onWindowResize);
  
  // UI Button Listeners
  const exportBtn = document.getElementById('export-glb-btn');
  if (exportBtn) exportBtn.addEventListener('click', exportToGLB);
  
  const captureBtn = document.getElementById('capture-btn');
  if (captureBtn) captureBtn.addEventListener('click', startCaptureCountdown);

  const rescanBtn = document.getElementById('rescan-btn');
  if (rescanBtn) rescanBtn.addEventListener('click', resetToScanMode);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// ==========================================================================
// 4. SCANNER STATE MACHINE & MESH UPDATE
// ==========================================================================

let appState = 'SCANNING'; // 'SCANNING' | 'COUNTDOWN' | 'VIEWING'
let faceUvs;
let staticTexture = null;
let savedLandmarks = null;
let orbitControls;
let scanProgress = 0;
let scanComplete = false;

function startCaptureCountdown() {
  if (appState !== 'SCANNING' || !state.metrics.faceDetected) return;
  appState = 'COUNTDOWN';
  
  const countdownOverlay = document.getElementById('countdown-overlay');
  const countdownText = document.getElementById('countdown-text');
  const captureUi = document.getElementById('capture-ui');
  const cameraFlash = document.getElementById('camera-flash');
  
  captureUi.classList.add('hidden');
  countdownOverlay.classList.remove('hidden');
  
  let count = 3;
  countdownText.innerText = count;
  
  const interval = setInterval(() => {
    count--;
    if (count > 0) {
      countdownText.innerText = count;
      countdownText.style.animation = 'none';
      void countdownText.offsetWidth; // trigger reflow
      countdownText.style.animation = 'pulse-countdown 1s infinite cubic-bezier(0.16, 1, 0.3, 1)';
    } else {
      clearInterval(interval);
      countdownOverlay.classList.add('hidden');
      
      // Flash effect
      cameraFlash.classList.add('flash-active');
      setTimeout(() => cameraFlash.classList.remove('flash-active'), 800);
      
      generate3DModel();
    }
  }, 1000);
}

function generate3DModel() {
  appState = 'VIEWING';
  
  // 1. Save current facial landmarks
  savedLandmarks = JSON.parse(JSON.stringify(state.metrics.lastMesh));
  
  // 2. Capture current webcam frame to CanvasTexture (Unmirrored for accurate UV match)
  const canvas = document.createElement('canvas');
  canvas.width = webcamEl.videoWidth || 640;
  canvas.height = webcamEl.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  
  // Enhance contrast slightly for a sharper, more detailed scan
  ctx.filter = 'contrast(1.08) saturate(1.05)';
  // Draw raw video frame (No flip needed, UVs handle mapping)
  ctx.drawImage(webcamEl, 0, 0, canvas.width, canvas.height);
  
  if (staticTexture) staticTexture.dispose();
  staticTexture = new THREE.CanvasTexture(canvas);
  staticTexture.encoding = THREE.sRGBEncoding;
  
  // Force sharp texture filtering (prevents blurry mipmap interpolation)
  staticTexture.minFilter = THREE.LinearFilter;
  staticTexture.magFilter = THREE.LinearFilter;
  staticTexture.generateMipmaps = false;
  if (renderer && renderer.capabilities) {
    staticTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }
  
  // 3. Update Material to Realistic Textured Model
  // Using StandardMaterial with the texture as a bump map gives incredible micro-detail 
  // (pores, wrinkles, stubble will actually catch light and cast tiny shadows!)
  const viewingMat = new THREE.MeshStandardMaterial({
    map: staticTexture,
    bumpMap: staticTexture, // Micro-details!
    bumpScale: 0.015,       // Subtle but sharp
    roughness: 0.65,
    metalness: 0.0,
    side: THREE.DoubleSide
  });
  
  // Adjust lights for realistic viewing (Turn off neon, use soft daylight)
  ambientLight.intensity = 1.0;
  ambientLight.color.set(0xffffff);
  keyLight.intensity = 0.6;
  keyLight.color.set(0xffffe6);
  keyLight.position.set(1, 2, 4);
  pointLightMagenta.visible = false;
  pointLightCyan.visible = false;
  pointLightBlue.visible = false;
  backLight.intensity = 0.3;
  
  faceMeshObject.material = viewingMat;
  
  // Reset Rotation
  faceMeshObject.rotation.set(0, 0, 0);
  sculpture.visible = false;
  
  // Enable OrbitControls
  orbitControls.enabled = true;
  camera.position.set(0, 0, 5); // Bring camera closer
  orbitControls.target.set(0, 0, 0);
  orbitControls.update();
  
  // Show Viewing UI
  document.getElementById('export-glb-btn').style.display = 'flex';
  document.getElementById('rescan-btn').style.display = 'flex';
}

function resetToScanMode() {
  appState = 'SCANNING';
  
  // Hide Viewing UI
  document.getElementById('export-glb-btn').style.display = 'none';
  document.getElementById('rescan-btn').style.display = 'none';
  
  // Reset Topology Scan Progress
  scanProgress = 0;
  scanComplete = false;
  document.getElementById('topology-scan-ui').classList.add('hidden');
  document.getElementById('capture-ui').classList.add('hidden');
  
  // Disable OrbitControls & Reset Camera
  orbitControls.enabled = false;
  camera.position.set(0, 0, 6.5);
  camera.lookAt(0, 0, 0);
  
  // Revert Material to Holographic Wireframe
  faceMeshObject.material = faceMeshMat;
  faceMeshMat.wireframe = true;
  faceMeshMat.color.set(0x00ffcc);
  faceMeshMat.map = null;
  faceMeshMat.transmission = 0.1;
  faceMeshMat.roughness = 0.2;
  faceMeshMat.metalness = 0.8;
  faceMeshMat.needsUpdate = true;
  
  // Revert Lights to Cyberpunk Rig
  ambientLight.intensity = 0.8;
  ambientLight.color.set(0xffffff);
  keyLight.intensity = 1.2;
  keyLight.color.set(0xffffff);
  keyLight.position.set(2, 4, 5);
  pointLightMagenta.visible = true;
  pointLightCyan.visible = true;
  pointLightBlue.visible = true;
  backLight.intensity = 0.5;
}

function updateDirectFaceMesh(landmarks) {
  // Compute face bounding box for normalization
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    if (pt[0] < minX) minX = pt[0];
    if (pt[0] > maxX) maxX = pt[0];
    if (pt[1] < minY) minY = pt[1];
    if (pt[1] > maxY) maxY = pt[1];
    if (pt[2] < minZ) minZ = pt[2];
    if (pt[2] > maxZ) maxZ = pt[2];
  }

  const cX = (minX + maxX) / 2;
  const cY = (minY + maxY) / 2;
  const cZ = (minZ + maxZ) / 2;
  const mSpan = Math.max(maxX - minX, maxY - minY) || 1;

  const vidW = webcamEl.videoWidth || 640;
  const vidH = webcamEl.videoHeight || 480;

  // Map all 468 landmarks to Three.js coordinates and dynamic UVs
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    
    // 3D Geometry Extrusion (using mSpan for all axes preserves true physical face proportions)
    faceVertexPositions[i * 3]     = -((pt[0] - cX) / mSpan) * 4.5;       // X: mirror & scale
    faceVertexPositions[i * 3 + 1] = -((pt[1] - cY) / mSpan) * 4.5;       // Y: flip & scale
    faceVertexPositions[i * 3 + 2] = -((pt[2] - cZ) / mSpan) * 4.5 + 0.5; // Z: true proportional depth
    
    // Dynamic Screen-Space UV Mapping (Fixed orientation)
    faceUvs[i * 2]     = pt[0] / vidW;
    faceUvs[i * 2 + 1] = 1.0 - (pt[1] / vidH);
  }

  faceMeshGeo.attributes.position.needsUpdate = true;
  faceMeshGeo.attributes.uv.needsUpdate = true;
  faceMeshGeo.computeVertexNormals();
}

/**
 * GLB Export Logic
 */
function exportToGLB() {
  if (appState !== 'VIEWING' || !faceMeshObject) return;
  
  const btn = document.getElementById('export-glb-btn');
  btn.innerHTML = '<span class="btn-text">BAKING...</span>';
  
  // Clone geometry and create an export-friendly physically based material
  const exportGeo = faceMeshGeo.clone();
  const exportMat = new THREE.MeshStandardMaterial({
    map: staticTexture,
    bumpMap: staticTexture,
    bumpScale: 0.015,
    roughness: 0.65,
    metalness: 0.0,
    side: THREE.DoubleSide
  });
  
  const exportMesh = new THREE.Mesh(exportGeo, exportMat);
  exportMesh.rotation.copy(faceMeshObject.rotation);
  
  // Export to GLB
  const exporter = new THREE.GLTFExporter();
  exporter.parse(exportMesh, function (glb) {
    const blob = new Blob([glb], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.style.display = 'none';
    link.href = url;
    link.download = 'neural_sculpture_face.glb';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    btn.innerHTML = '<span class="btn-icon">⬇</span><span class="btn-text">EXPORT .GLB</span>';
  }, { binary: true });
}

/**
 * Deform the idle icosahedron sphere with breathing/mouse effects.
 * This sphere fades out when a face is detected and the face mesh takes over.
 */
function updateSphereDeformation(time) {
  const positionAttr = geometry.attributes.position;
  const posArray = positionAttr.array;

  // FALLBACK CONTROLS: If face is not detected, use mouse inputs!
  if (!state.metrics.faceDetected) {
    state.metrics.yaw = mouseX * 1.3;
    state.metrics.pitch = -mouseY * 1.1;
    state.metrics.roll = mouseX * mouseY * 0.4;
    state.metrics.mouthOpen = isMouseDown ? 0.9 : 0.0;
    state.metrics.smileWidth = isMouseDown ? 0.9 : 0.15;
    state.metrics.isBlinking = isMouseDown;
  }

  // LERP input parameters
  const lerpFactor = 0.12;
  state.smoothed.mouthOpen += (state.metrics.mouthOpen - state.smoothed.mouthOpen) * lerpFactor;
  state.smoothed.smileWidth += (state.metrics.smileWidth - state.smoothed.smileWidth) * lerpFactor;
  const targetBlink = state.metrics.isBlinking ? 1.0 : 0.0;
  state.smoothed.blink += (targetBlink - state.smoothed.blink) * 0.18;
  state.smoothed.yaw += (state.metrics.yaw - state.smoothed.yaw) * lerpFactor;
  state.smoothed.pitch += (state.metrics.pitch - state.smoothed.pitch) * lerpFactor;
  state.smoothed.roll += (state.metrics.roll - state.smoothed.roll) * lerpFactor;

  // Face morph factor — controls crossfade between sphere and face mesh
  const targetMorph = state.metrics.faceDetected ? 1.0 : 0.0;
  state.smoothed.faceMorph += (targetMorph - state.smoothed.faceMorph) * 0.1;

  // Crossfade: sphere fades completely out, face mesh fades in
  material.opacity = 1.0 - state.smoothed.faceMorph;
  material.transparent = true;
  faceMeshMat.opacity = state.smoothed.faceMorph;

  // Fully hide the idle sphere when completely faded out to prevent depth sorting/blocking issues
  sculpture.visible = material.opacity > 0.01;
  faceMeshObject.visible = faceMeshMat.opacity > 0.01;

  // Apply rotation to BOTH meshes
  const rotY = state.smoothed.yaw * 0.8;
  const rotX = state.smoothed.pitch * 0.8;
  const rotZ = -state.smoothed.roll * 0.7;

  sculpture.rotation.set(rotX, rotY, rotZ);
  faceMeshObject.rotation.set(rotX, rotY, rotZ);

  // Blink ripple intensity
  const ripplePower = 0.05 + state.smoothed.blink * 0.35;

  // Deform icosahedron sphere with organic effects
  for (let i = 0; i < posArray.length; i += 3) {
    const bx = originalPositions[i];
    const by = originalPositions[i + 1];
    const bz = originalPositions[i + 2];

    const r = Math.sqrt(bx * bx + by * by + bz * bz);
    const nx = bx / r;
    const ny = by / r;
    const nz = bz / r;

    let displacement = 0;

    // Organic breathing waves
    displacement += Math.sin(bx * 1.5 + time * 2.0) * Math.cos(by * 1.5 + time * 1.8) * 0.08;

    // Mouth cavity
    const mDist = Math.sqrt(bx * bx + (by + 0.6) * (by + 0.6) + (bz - 1.8) * (bz - 1.8));
    if (mDist < 1.4) {
      displacement -= state.smoothed.mouthOpen * ((1.4 - mDist) / 1.4) * 1.1;
    }

    // Ripple waves
    const dTop = Math.sqrt(bx * bx + (by - 2.0) * (by - 2.0) + bz * bz);
    displacement += Math.sin(dTop * 6.5 - time * 8.5) * ripplePower * 0.16;

    // Smile stretch
    const smileStretch = 1.0 + (state.smoothed.smileWidth - 0.5) * 0.35;

    posArray[i] = bx * smileStretch + nx * displacement;
    posArray[i + 1] = by + ny * displacement;
    posArray[i + 2] = bz + nz * displacement;
  }

  positionAttr.needsUpdate = true;
  geometry.computeVertexNormals();
}

function animate() {
  requestAnimationFrame(animate);

  const time = performance.now() * 0.001;

  // 0. Update Webcam Preview Canvas
  if (state.cameraActive) {
    updateWebcamPreview();
  }

  // Orbit colored lights
  if (pointLightMagenta) {
    pointLightMagenta.position.x = Math.sin(time * 0.8) * 4.5;
    pointLightMagenta.position.z = Math.cos(time * 0.8) * 4.5;
  }
  if (pointLightCyan) {
    pointLightCyan.position.x = -Math.sin(time * 1.1) * 4.5;
    pointLightCyan.position.z = -Math.cos(time * 1.1) * 4.5;
  }
  if (pointLightBlue) {
    pointLightBlue.position.y = Math.sin(time * 0.6) * 3 + 2;
  }

  // Update logic based on App State
  if (appState === 'SCANNING' || appState === 'COUNTDOWN') {
    if (state.metrics.faceDetected && state.metrics.lastMesh) {
      updateDirectFaceMesh(state.metrics.lastMesh);
      
      // Progress the topology scan only in SCANNING mode
      if (appState === 'SCANNING') {
        if (!scanComplete) {
          scanProgress += 1.0; // Reaches 100 in ~100 frames (~1.6 seconds)
          if (scanProgress >= 100) {
            scanProgress = 100;
            scanComplete = true;
            document.getElementById('topology-scan-ui').classList.add('hidden');
            document.getElementById('capture-ui').classList.remove('hidden');
          } else {
            document.getElementById('topology-scan-ui').classList.remove('hidden');
            document.getElementById('capture-ui').classList.add('hidden');
            document.getElementById('topology-bar-fill').style.width = scanProgress + '%';
            document.getElementById('topology-scan-text').innerText = `ANALYZING FACIAL TOPOLOGY... ${Math.floor(scanProgress)}%`;
          }
        }
      }
    } else if (appState === 'SCANNING') {
      // Face lost, reset scan
      scanProgress = 0;
      scanComplete = false;
      document.getElementById('topology-scan-ui').classList.add('hidden');
      document.getElementById('capture-ui').classList.add('hidden');
    }
    
    // Deform idle sphere + handle crossfade
    if (geometry && originalPositions) {
      updateSphereDeformation(time);
    }
  } else if (appState === 'VIEWING') {
    if (orbitControls && orbitControls.enabled) {
      orbitControls.update();
    }
  }

  // Render Three.js
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }

  // Update audio synth
  if (audioInitialized) {
    updateSynthesizerParams();
  }

  // Telemetry FPS
  updateTelemetry();
}

// ==========================================================================
// 5. INTERACTIVE AUDIO SYNTHESIZER (WEB AUDIO API)
// ==========================================================================
function initAudio() {
  if (audioInitialized) return;

  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
    masterGain = audioCtx.createGain();
    masterGain.gain.setValueAtTime(0.0, audioCtx.currentTime);
    masterGain.connect(audioCtx.destination);
    
    // Smooth master audio fade-in
    masterGain.gain.linearRampToValueAtTime(0.35, audioCtx.currentTime + 2.0);

    // Sub Drone Oscillator
    droneOsc = audioCtx.createOscillator();
    droneOsc.type = 'sawtooth';
    droneOsc.frequency.setValueAtTime(75.0, audioCtx.currentTime);

    // Warm deep triangle sub-bass
    subOsc = audioCtx.createOscillator();
    subOsc.type = 'triangle';
    subOsc.frequency.setValueAtTime(37.5, audioCtx.currentTime);

    // Resonant sweep lowpass filter
    synthFilter = audioCtx.createBiquadFilter();
    synthFilter.type = 'lowpass';
    synthFilter.Q.setValueAtTime(6.0, audioCtx.currentTime);
    synthFilter.frequency.setValueAtTime(350, audioCtx.currentTime);

    // LFO modulator
    lfo = audioCtx.createOscillator();
    lfo.type = 'sine';
    lfo.frequency.setValueAtTime(1.5, audioCtx.currentTime);

    lfoGain = audioCtx.createGain();
    lfoGain.gain.setValueAtTime(80, audioCtx.currentTime);

    lfo.connect(lfoGain);
    lfoGain.connect(synthFilter.frequency);

    const oscMixer = audioCtx.createGain();
    oscMixer.gain.setValueAtTime(0.65, audioCtx.currentTime);
    droneOsc.connect(oscMixer);
    
    const subMixer = audioCtx.createGain();
    subMixer.gain.setValueAtTime(0.7, audioCtx.currentTime);
    subOsc.connect(subMixer);

    oscMixer.connect(synthFilter);
    synthFilter.connect(masterGain);
    subMixer.connect(masterGain);

    droneOsc.start();
    subOsc.start();
    lfo.start();

    audioInitialized = true;
    console.log("Audio feedback loop initialized successfully.");
    audioShield.classList.add('hidden');
  } catch (err) {
    console.warn("Web Audio API not supported on this browser.", err);
  }
}

function updateSynthesizerParams() {
  if (!audioCtx || audioCtx.state === 'suspended') return;

  const now = audioCtx.currentTime;

  // Mouth open modulates pitch lower (closed = 75Hz, open = 45Hz)
  const baseFreq = 75.0 - (state.smoothed.mouthOpen * 30.0);
  droneOsc.frequency.setTargetAtTime(baseFreq, now, 0.1);
  subOsc.frequency.setTargetAtTime(baseFreq / 2, now, 0.1);

  // Smile width opens lowpass filter cutoff
  const cutoff = 250 + (state.smoothed.smileWidth * 1600);
  synthFilter.frequency.setTargetAtTime(cutoff, now, 0.15);

  // Blink ripple speeds up LFO frequency
  const lfoFreq = 1.5 + (state.smoothed.blink * 10.0);
  lfo.frequency.setTargetAtTime(lfoFreq, now, 0.1);

  const lfoDepth = 80 + (state.smoothed.blink * 350);
  lfoGain.gain.setTargetAtTime(lfoDepth, now, 0.1);
}

document.body.addEventListener('click', () => {
  if (!audioInitialized) {
    initAudio();
  } else if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
    audioShield.classList.add('hidden');
  }
});

// ==========================================================================
// 6. TELEMETRY & BOOTSTRAP
// ==========================================================================
function updateTelemetry() {
  state.frameCount++;
  const now = performance.now();

  if (now - state.fpsIntervalTime >= 500) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.fpsIntervalTime));
    fpsCounter.textContent = state.fps.toString().padStart(2, '0');
    state.frameCount = 0;
    state.fpsIntervalTime = now;
  }

  // Render Status text based on tracking states
  if (state.metrics.faceDetected) {
    trackingStatus.textContent = "AI ACTIVE";
    trackingStatus.className = "value active";
  } else if (state.cameraActive && state.modelLoaded) {
    trackingStatus.textContent = "SEARCHING";
    trackingStatus.className = "value searching";
  } else if (state.modelLoaded) {
    trackingStatus.textContent = "MOUSE MODE";
    trackingStatus.className = "value active";
    trackingStatus.style.color = "var(--neon-cyan)";
  } else {
    trackingStatus.textContent = "COMPILING";
    trackingStatus.className = "value searching";
  }
}

// Bootstrap Application
async function bootstrap() {
  try {
    updateLoadingProgress(10, "SPINNING UP 3D ENVIRONMENT // INITIALIZING WEBGL ENGINE");
    
    // 1. Instantly boot the ThreeJS WebGL engine
    initThreeEngine();
    updateLoadingProgress(20, "THREE.JS ENGINE COMPILED // RECRUITING Rim Light SHIMMER RIGS");
    
    // Start animation loop rendering the interactive sculpture
    animate();
    
    // 2. Trigger asynchronous background hardware hooks & ML loads
    setTimeout(() => {
      initBackgroundServices();
    }, 400); // 400ms delay for smooth aesthetic progression
  } catch (err) {
    console.error("Critical error in engine bootstrap:", err);
    finalizeLoading(false, "CRITICAL GRAPHICS ERROR // DIRECT TO MOUSE CONTROLLER");
  }
}

function checkOriginSecurity() {
  const isInsecure = window.location.protocol === 'file:' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia;
  if (isInsecure) {
    console.warn("Insecure origin (file://) or missing mediaDevices detected. Injecting safety advisory banner.");
    const banner = document.createElement('div');
    banner.id = 'security-warning-banner';
    banner.style.cssText = `
      position: absolute;
      bottom: 110px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255, 0, 127, 0.12);
      border: 1px solid rgba(255, 0, 127, 0.35);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-radius: 16px;
      padding: 14px 28px;
      color: #ffccd8;
      font-family: 'Outfit', sans-serif;
      font-size: 0.75rem;
      letter-spacing: 0.8px;
      z-index: 1000;
      text-align: center;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(255, 0, 127, 0.15);
      max-width: 90%;
      width: 480px;
      line-height: 1.6;
      pointer-events: auto;
      transition: all 0.3s ease;
    `;
    banner.innerHTML = `
      <span style="font-size: 1.1rem; margin-right: 6px; vertical-align: middle;">⚠️</span>
      <strong style="color: #ff4d94; letter-spacing: 1.5px;">INSECURE CONTEXT (FILE://) DETECTED</strong><br>
      Browsers completely block camera hardware requests on local disk file folders.<br>
      To activate real-time facial sculpting, please open the live local server:<br>
      <a href="http://localhost:8080" target="_blank" style="color: #00ffcc; text-decoration: none; font-weight: 800; border-bottom: 1px dashed #00ffcc; margin-top: 6px; display: inline-block; letter-spacing: 1px;">http://localhost:8080</a>
    `;
    document.body.appendChild(banner);
  }
}

async function initBackgroundServices() {
  try {
    // Check if security context blocks mediaDevices
    checkOriginSecurity();

    if (typeof tf === 'undefined' || typeof faceLandmarksDetection === 'undefined') {
      debugLog('❌ CDN scripts missing — tf:' + (typeof tf) + ' faceLandmarks:' + (typeof faceLandmarksDetection));
      state.modelLoaded = true;
      finalizeLoading(false, "CDN SCRIPTS BLOCKED // ACTIVATING CYBER MOUSE MODE");
      return;
    }
    debugLog('✅ CDN libraries loaded (tf + faceLandmarksDetection)');

    updateLoadingProgress(30, "INITIALIZING CYBERNETIC EYE // REQUESTING WEBCAM DEVICE");

    // 2. Load webcam
    const camSuccess = await initWebcam();
    if (camSuccess) {
      debugLog('Camera OK → loading FaceMesh AI model...');
      updateLoadingProgress(50, "WEBCAM SIGNAL ACTIVE // INITIALIZING ML LANDMARK INTERPRETER");
      // If camera connected, trigger FaceMesh load
      await initFaceMesh();
    } else {
      debugLog('⚠️ Webcam unavailable — mouse fallback mode');
      state.modelLoaded = true; // Sets HUD state to MOUSE MODE
      finalizeLoading(false, "WEBCAM SIGNAL OFFLINE // ACTIVATING CYBER MOUSE MODE");
    }
  } catch (err) {
    console.warn("Background service startup issue. Continuing in Mouse Interaction mode.", err);
    state.modelLoaded = true;
    finalizeLoading(false, "HARDWARE INITIALIZATION ISSUE // DIRECT TO MOUSE CONTROLLER");
  }
}

// Start
window.onload = bootstrap;
