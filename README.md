# On-Device Activity Classifier (Expo + TensorFlow Lite)

This project demonstrates how to train a machine-learning model on accelerometer data and deploy it on a mobile phone for **real-time, offline inference** using React Native, Expo Dev Client, and TensorFlow Lite.

The model runs entirely on the device without sending any data to a server.

---

## 1. Project Overview

- Platform: React Native (Expo)
    
- Sensors: Android Accelerometer
    
- ML Framework (training): TensorFlow (Python)
    
- ML Runtime (mobile): TensorFlow Lite
    
- Inference: On-device, offline
    
- Build system: Expo Dev Client + Android Studio
    

---

## 2. Initial App Setup (Expo)

1. Create an Expo project:
    
    `npx create-expo-app ActivityClassifier cd ActivityClassifier`
    
2. Run the app with Expo Go to confirm setup:
    
    `npx expo start`
    

At this stage, the app is JavaScript-only and cannot run native ML code.

---

## 3. Sensor Data Collection (Accelerometer)

Inside the main screen:

- Used `expo-sensors` to access the accelerometer
    
- Collected `(x, y, z)` values at ~20 Hz
    
- Maintained a rolling buffer of the last 128 samples
    
- Added UI buttons to:
    
    - Start / Stop recording
        
    - Record labeled windows (Still, Walking, Running)
        

Each labeled sample stored:

- 128 accelerometer samples
    
- Activity label
    
- Timestamp
    

Data was saved locally as:

`activity_dataset.json`

This file was exported from the phone and used for training.

---

## 4. Model Training (Python + TensorFlow)

### Environment

- Python 3.10 (virtual environment)
    
- TensorFlow 2.15
    

### Training Steps

1. Load `activity_dataset.json`
    
2. Convert samples into:
    
    - Input shape: `(128, 3)`
        
    - Labels: Still / Walking / Running
        
3. Train a simple neural network classifier
    
4. Save the trained model as a TensorFlow SavedModel
    

Output:

`saved_model_activity/`

---

## 5. Convert TensorFlow Model to TensorFlow Lite

Using TensorFlow Lite for mobile inference:

`import tensorflow as tf  converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_activity") converter.optimizations = [tf.lite.Optimize.DEFAULT] tflite_model = converter.convert()  with open("activity.tflite", "wb") as f:     f.write(tflite_model)`

Output:

`activity.tflite`

---

## 6. Add Native Support (Expo Dev Client)

TensorFlow Lite requires native code, so Expo Go cannot be used.

Steps:

1. Install dev client:
    
    `npx expo install expo-dev-client`
    
2. Prebuild native project:
    
    `npx expo prebuild`
    
3. Build and install app:
    
    `npx expo run:android`
    

This generated an Android app with native modules enabled.

---

## 7. Install TensorFlow Lite Runtime

Installed a native TFLite binding for React Native:

`npm install react-native-fast-tflite npx expo prebuild --clean npx expo run:android`

This library provides:

- Model loading
    
- Synchronous inference
    
- Access to output tensors
    

---

## 8. Bundle the TFLite Model

1. Place the model inside the project:
    
    `assets/images/models/activity.tflite`
    
2. Configure Metro to allow `.tflite` assets:
    

`metro.config.js`

`const { getDefaultConfig } = require("expo/metro-config");  const config = getDefaultConfig(__dirname); config.resolver.assetExts.push("tflite");  module.exports = config;`

3. Restart Metro:
    
    `npx expo start -c`
    

---

## 9. Load the Model in React Native

In the main screen:

`import { loadTensorflowModel } from "react-native-fast-tflite";  const model = await loadTensorflowModel(   require("../../assets/images/models/activity.tflite") );`

A successful load confirms that the model is available on the device.

---

## 10. Run Real-Time Inference

### Inference Pipeline

1. Collect accelerometer samples continuously
    
2. Keep the latest 128 samples
    
3. Every ~1 second:
    
    - Flatten samples into a `Float32Array`
        
    - Run inference:
        
        `const outputs = model.runSync([input]);`
        
    - Select highest probability output
        
    - Map index to label (Still / Walking / Running)
        
4. Update UI with predicted activity and confidence
    

The model now runs **fully offline** on the phone.

---

## 11. Result

- The app correctly detects:
    
    - Still
        
    - Walking
        
    - Running
        
- Inference happens locally
    
- No internet connection required
    
- No sensor data leaves the device
    

---

## 12. What This Demonstrates

- End-to-end ML pipeline
    
- Real sensor data collection
    
- Model training and optimization
    
- Native mobile inference
    
- Privacy-preserving, offline AI
    

This is the same architecture used in real fitness, health, and activity-tracking applications.

---

## 13. Possible Extensions

- Prediction smoothing / majority voting
    
- Activity history logging
    
- Background tracking
    
- Fall detection
    
- BLE health device integration
    
- Medical or fitness dashboards
    

---

## 14. Summary

This project shows how to go from:

- Raw sensor data  
    to
    
- A trained ML model  
    to
    
- On-device inference in a mobile app
    

without relying on cloud services or external APIs.