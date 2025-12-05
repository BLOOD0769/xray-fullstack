import React, { useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Map model indices to human readable labels
const CLASS_MAP = {
  0: "Normal",
  1: "Pneumonia"
};

function formatPercent(p) {
  // p is 0..1 -> return rounded percent string, e.g. "94%"
  return `${Math.round(p * 100)}%`;
}

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [overlay, setOverlay] = useState(null);

  function handleFileSelect(e) {
    setFile(e.target.files[0]);
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return alert("Please upload an image first!");

    setLoading(true);
    setPrediction(null);
    setOverlay(null);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/predict`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // res.data.probs is expected as [p_normal, p_pneumonia]
      const probs = res.data.probs || [];
      const predIdx = Number(res.data.pred_idx ?? (probs.length ? probs.indexOf(Math.max(...probs)) : 0));
      const labelName = CLASS_MAP[predIdx] ?? String(predIdx);

      // build a small structured prediction object for the UI
      const formatted = {
        pred_idx: predIdx,
        label: labelName,
        probs: probs.map(p => Number(p)) // ensure numeric
      };

      setPrediction(formatted);
      if (res.data.overlay_b64) {
        setOverlay(`data:image/jpeg;base64,${res.data.overlay_b64}`);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      alert("Error: " + msg);
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <label htmlFor="file-input">
        <div className="upload-box">
          <p style={{ fontSize: "18px", marginBottom: "8px" }}>
            Click to upload X-Ray image
          </p>
          <p style={{ color: "#6b7280" }}>
            PNG, JPG, or JPEG — max recommended size: 5MB
          </p>
        </div>
      </label>

      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: "none" }}
      />

      {file && <p style={{ marginTop: "10px" }}>Selected: {file.name}</p>}

      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Analyze X-Ray"}
      </button>

      {loading && <div className="spinner"></div>}

      {prediction && (
        <div className="result-card">
          <h3>Probable condition</h3>

          <p>
            <strong>{prediction.label}</strong>
          </p>

          <p style={{ marginTop: 8 }}>
            <strong>Confidence Scores (rounded):</strong>
          </p>

          <div style={{ background: "#f3f4f6", padding: 12, borderRadius: 8 }}>
            <div><strong>{formatPercent(prediction.probs[0] ?? 0)}</strong> — {CLASS_MAP[0]}</div>
            <div><strong>{formatPercent(prediction.probs[1] ?? 0)}</strong> — {CLASS_MAP[1]}</div>
          </div>

          <p style={{ marginTop: 12, color: "#374151" }}>
            Explanation: {formatPercent(prediction.probs[0] ?? 0)} refers to <strong>{CLASS_MAP[0]}</strong> and {formatPercent(prediction.probs[1] ?? 0)} refers to <strong>{CLASS_MAP[1]}</strong>.
          </p>

          <h4 style={{ marginTop: "20px" }}>Grad-CAM Heatmap</h4>
          {overlay && (
            <>
              <img
                src={overlay}
                alt="GradCAM Overlay"
                className="image-preview"
                style={{ maxWidth: "100%", borderRadius: 12 }}
              />

              {/* Legend */}
              <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <LegendItem color="#2563eb" label="Blue → low importance" />
                  <LegendItem color="#38bdf8" label="Green → medium importance" />
                  <LegendItem color="#facc15" label="Yellow → high importance" />
                  <LegendItem color="#ef4444" label="Red → highest importance" />
                </div>
                <div style={{ marginLeft: "auto", color: "#6b7280", fontSize: 13 }}>
                  Note: Heatmap colors indicate where the model focused. Red/yellow = more important.
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function LegendItem({ color, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 22,
        height: 12,
        borderRadius: 3,
        background: color,
        boxShadow: "0 1px 3px rgba(0,0,0,0.15)"
      }} />
      <div style={{ fontSize: 14 }}>{label}</div>
    </div>
  );
}
