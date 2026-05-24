# 🧠 Neural Sculptor

**Neural Sculptor** is a real-time, browser-based AI facial topology engine that transforms a live webcam feed into an interactive, 3D WebGL sculpture. Using advanced machine learning models (TensorFlow.js and MediaPipe), the application maps 468 precise facial landmarks, constructing a live digital twin that responds perfectly to your facial expressions.

When ready, users can scan their facial geometry to generate a highly detailed, photorealistic 3D `.glb` model, perfectly mapped with micro-detail textures derived from the webcam feed, ready for export to Blender, Unity, or other 3D software.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## ✨ Features

- **Live Facial Tracking**: High-performance, 468-point facial landmark tracking running entirely client-side using TensorFlow.js and MediaPipe.
- **Cyberpunk Holographic Mode**: Idle mode features a dynamic, glassmorphic wireframe mesh illuminated by a real-time RGB neon lighting rig.
- **Photorealistic Scanning**: Uses high-definition canvas capture combined with physical bump mapping to translate raw webcam pixels into micro-detailed 3D skin textures (pores, wrinkles, and shadows).
- **One-Click GLB Export**: Bake your facial topology and physical materials into a standard `.glb` format, completely ready for any game engine or 3D modeling tool.
- **Dynamic Audio Synthesizer**: Built-in WebAudio API synthesizer featuring FM modulation and LFO low-pass filters that morph dynamically based on the vertical and horizontal movements of your jaw.
- **Graceful Failback**: Don't have a webcam? The app seamlessly transitions into a fallback "Mouse Mode" where cursor movements warp the mesh.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🚀 Getting Started

Neural Sculptor runs entirely in the browser and requires no server-side processing for the AI models. 

### Prerequisites
You will need a local development server to serve the files, as modern browsers block WebGL and Webcam permissions for `file://` URLs.

### Installation

1. Clone or download this repository to your local machine.
2. Navigate to the project directory:
   ```bash
   cd ai-facemesh-sculptor
   ```
3. Start a local HTTP server. If you have Python installed, you can run:
   ```bash
   python -m http.server 8080
   ```
   *Alternatively, use Node.js `http-server`, VS Code Live Server, or any other static file server.*
4. Open your browser and navigate to `http://localhost:8080`.
5. Grant the browser permission to access your webcam when prompted.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🕹️ Usage

1. **Wait for Compilation**: On first load, the TensorFlow engine will compile WebGL shaders and download the neural mesh weights.
2. **Holographic Tracking**: Once loaded, your face will appear as a neon wireframe. Move your head, open your mouth, and blink to see the mesh react in real-time.
3. **Audio Synthesizer**: Click anywhere on the screen to activate the audio engine. Open your mouth to modulate the synthesizer!
4. **Capture**: Click the **CAPTURE & GENERATE MODEL** button. Hold still for 3 seconds while the system analyzes your facial topology.
5. **View & Export**: A photorealistic, highly detailed 3D model of your face will be generated. Use your mouse to rotate and inspect the model. Click **EXPORT .GLB** to download the 3D asset, or click **RE-SCAN** to try again.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🛠️ Technology Stack

- **HTML5 & CSS3**: Custom styling, glassmorphism UI, and smooth CSS transitions.
- **JavaScript (ES6+)**: Core application logic and async model loading.
- **Three.js (WebGL)**: 3D scene rendering, dynamic lighting, custom shaders, and `.glb` exportation.
- **TensorFlow.js**: Machine learning backend utilizing WebGL acceleration.
- **MediaPipe FaceMesh**: Pre-trained ML model providing dense 468-point 3D face geometry.
- **WebAudio API**: Real-time sound synthesis tied to facial coordinate deltas.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📝 License

This project is open-source and free to use for non-commercial and educational purposes. 
