import { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { Accelerometer } from "expo-sensors";
import { loadTensorflowModel } from "react-native-fast-tflite";

type Accel = { x: number; y: number; z: number };

const LABELS = ["Still", "Walking", "Running"] as const;

// Must match what you trained on:
const WINDOW = 128; // samples
const FEATURES = 3; // x,y,z

export default function Index() {
  const subRef = useRef<any>(null);
  const bufferRef = useRef<Accel[]>([]);
  const modelRef = useRef<any>(null);
  const inferTimerRef = useRef<any>(null);

  const [running, setRunning] = useState(false);
  const [latest, setLatest] = useState<Accel | null>(null);
  const [activity, setActivity] = useState("—");
  const [confidence, setConfidence] = useState("—");
  const [motionScore, setMotionScore] = useState("—");
  const [modelStatus, setModelStatus] = useState("Loading model...");

  // 1) Load TFLite model once
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // IMPORTANT: path should point to your bundled .tflite
        const model = await loadTensorflowModel(
          require("../../assets/images/models/activity.tflite")
        );
        if (cancelled) return;
        modelRef.current = model;
        setModelStatus("Model loaded ✅");
      } catch (e: any) {
        setModelStatus("Model load failed ❌");
        console.log("Model load error:", e);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const start = () => {
    if (running) return;

    Accelerometer.setUpdateInterval(50); // ~20 Hz
    subRef.current = Accelerometer.addListener((data) => {
      setLatest(data);

      bufferRef.current.push(data);
      if (bufferRef.current.length > WINDOW) bufferRef.current.shift();

      // simple motion score (optional)
      const buf = bufferRef.current;
      if (buf.length >= 20) {
        const last = buf.slice(-20);
        const avgMag =
          last.reduce(
            (sum, s) => sum + Math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z),
            0
          ) / last.length;
        setMotionScore(avgMag.toFixed(3));
      }
    });

    // 2) Run inference every ~1s (not on every sensor tick)
    inferTimerRef.current = setInterval(runInference, 1000);

    setRunning(true);
  };

  const stop = () => {
    subRef.current?.remove?.();
    subRef.current = null;
    if (inferTimerRef.current) clearInterval(inferTimerRef.current);
    inferTimerRef.current = null;
    setRunning(false);
  };

  const runInference = () => {
    const model = modelRef.current;
    const buf = bufferRef.current;

    if (!model) {
      setConfidence("model not loaded");
      return;
    }
    if (buf.length < WINDOW) {
      setConfidence(`need ${WINDOW - buf.length} more samples`);
      return;
    }

    // 3) Build Float32 input [1, 128, 3] flattened
    // NOTE: This must match how you trained. If you normalized in training,
    // you MUST normalize here the same way.
    const win = buf.slice(-WINDOW);
    const input = new Float32Array(WINDOW * FEATURES);

    for (let i = 0; i < WINDOW; i++) {
      const s = win[i];
      input[i * 3 + 0] = s.x;
      input[i * 3 + 1] = s.y;
      input[i * 3 + 2] = s.z;
    }

    try {
      // runSync is fast; returns an array of outputs
      const outputs = model.runSync([input]);

      // most simple classifiers output a single tensor like [1,3] or [3]
      const probs = outputs[0] as Float32Array;

      // argmax
      let bestIdx = 0;
      let bestVal = probs[0] ?? 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > bestVal) {
          bestVal = probs[i];
          bestIdx = i;
        }
      }

      setActivity(LABELS[bestIdx] ?? "—");
      setConfidence(bestVal.toFixed(2));
    } catch (e: any) {
      setConfidence("inference error");
      console.log("Inference error:", e);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>On-Device Activity Classifier</Text>
      <Text style={{ marginTop: -8 }}>{modelStatus}</Text>

      <View style={styles.card}>
        <Text style={styles.label}>Status</Text>
        <Text>{running ? "Running" : "Stopped"}</Text>
        <Text>Motion score: {motionScore}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Latest Accelerometer</Text>
        <Text>
          {latest
            ? `x: ${latest.x.toFixed(2)}  y: ${latest.y.toFixed(2)}  z: ${latest.z.toFixed(2)}`
            : "—"}
        </Text>
        <Text>Buffer size: {bufferRef.current.length}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Prediction (TFLite)</Text>
        <Text style={styles.pred}>{activity}</Text>
        <Text>Confidence: {confidence}</Text>
      </View>

      <View style={styles.row}>
        <TouchableOpacity style={styles.btn} onPress={start}>
          <Text style={styles.btnText}>Start</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.btn} onPress={stop}>
          <Text style={styles.btnText}>Stop</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 16 },
  title: { fontSize: 22, fontWeight: "700" },
  card: { padding: 14, borderRadius: 10, backgroundColor: "#f2f2f2" },
  label: { fontWeight: "600", marginBottom: 4 },
  pred: { fontSize: 18, fontWeight: "700" },
  row: { flexDirection: "row", gap: 12 },
  btn: {
    flex: 1,
    padding: 14,
    borderRadius: 10,
    backgroundColor: "#007AFF",
    alignItems: "center",
  },
  btnText: { color: "#fff", fontWeight: "600" },
});
