import React from "react";
import UploadPredict from "./components/UploadPredict";

export default function App() {
  return (
    <>
      <h1>Chest X-Ray Analyzer</h1>

      <div className="container">
        <UploadPredict />
      </div>
    </>
  );
}
