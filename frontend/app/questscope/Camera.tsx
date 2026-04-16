"use client";

import { useEffect, useRef, useState } from "react";
import { initFaceDetection } from "@/lib/faceDetection";
import {
  createInitialStats,
  processFrameValidity,
  evaluateVideoQuality,
  ValidityStats,
} from "@/lib/validity";

import {
  initPose,
  sendPoseFrame,
  getLatestPose,
} from "@/lib/poseExtraction";

import {
  initFaceLandmarker,
  sendFaceLandmarkFrame,
  getLatestFaceLandmarks,
} from "@/lib/faceLandmarks";

import { extractHeadPose } from "@/lib/headPose";
import { estimateGaze } from "@/lib/gaze";

type SequenceFrame = {
  valid: boolean;
  skeleton: Record<string, number[]> | null;
  eye_gaze: Record<string, number | null> | null;
  head_gaze: Record<string, number | null> | null; // Keeps original head pitch/yaw mapping for DREAM
};

export default function Camera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [status, setStatus] = useState("Initializing...");
  const [result, setResult] = useState("");

  useEffect(() => {
    let animationId: number;

    let stats: ValidityStats = createInitialStats();
    let validMask: boolean[] = [];
    let sequence: SequenceFrame[] = [];

    const RECORDING_DURATION_MS = 10000;
    // TARGET_FPS is the max we'll try to capture — real FPS will be lower
    // and is measured accurately at the end. Min acceptable: 7 FPS.
    const TARGET_FPS = 15;
    const MIN_FPS = 7;
    const FRAME_INTERVAL = 1000 / TARGET_FPS;

    let lastFrameTime = 0;
    let startTime = 0;
    let firstFrameTime = 0; // track when first frame actually landed

    async function setup() {
      await initFaceDetection();
      await initPose();
      await initFaceLandmarker();

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
        });

      if (!videoRef.current) return;

      videoRef.current.srcObject = stream;

      await new Promise<void>((resolve) => {
        videoRef.current!.onloadedmetadata = () => resolve();
      });

      await videoRef.current.play();

      while (videoRef.current.readyState < 2) {
        await new Promise((r) => setTimeout(r, 50));
      }

      setStatus("Recording...");
      startTime = performance.now();
      firstFrameTime = 0; // reset; set on first frame capture

      loop();
    }

    function loop() {
      if (!videoRef.current || !canvasRef.current) {
        animationId = requestAnimationFrame(loop);
        return;
      }

      const now = performance.now();

      if (now - lastFrameTime >= FRAME_INTERVAL) {
        lastFrameTime = now;

        // Track when first frame actually lands (for accurate FPS)
        if (firstFrameTime === 0) firstFrameTime = now;

        // ---- STEP 1: VALIDITY ----
        processFrameValidity(
          videoRef.current,
          canvasRef.current,
          stats,
          validMask
        );

        const isValid =
          validMask.length > 0
            ? validMask[validMask.length - 1]
            : false;

        // ---- STEP 2: BODY POSE (SKELETON) ----
        sendPoseFrame(videoRef.current);
        const skeleton = isValid ? getLatestPose() : null;

        // ---- STEP 3: FACE LANDMARKS ----
        sendFaceLandmarkFrame(videoRef.current);
        const faceMesh = getLatestFaceLandmarks();

        // ---- STEP 4: HEAD GAZE ----
        const head_gaze = isValid
          ? extractHeadPose(faceMesh, isValid)
          : null;

        // ---- STEP 5: EYE GAZE ----
        const eye_gaze = isValid ? estimateGaze(faceMesh) : null;

        // ---- STORE FRAME ----
        sequence.push({
          valid: isValid,
          skeleton: isValid ? skeleton : null,
          head_gaze: isValid ? head_gaze : null,
          eye_gaze: isValid ? eye_gaze : null,
        });
      }

      if (now - startTime >= RECORDING_DURATION_MS) {
        // Measure FPS from the actual span between first and last frame,
        // not the nominal recording duration — more accurate under load.
        const elapsedSec =
          firstFrameTime > 0
            ? (now - firstFrameTime) / 1000
            : RECORDING_DURATION_MS / 1000;

        const actualFps = sequence.length / elapsedSec;

        const decision = evaluateVideoQuality(
          validMask,
          stats,
          Math.round(MIN_FPS * (RECORDING_DURATION_MS / 1000)), // min valid frames
          0.7,
          3.0,
          actualFps
        );

        console.log(
          `Recording done: ${sequence.length} frames @ ${actualFps.toFixed(1)} FPS | ${decision.reason}`
        );

        // Reject if FPS was too low for meaningful inference
        if (actualFps < MIN_FPS) {
          setStatus("Finished");
          setResult(
            `Failed: FPS too low (${actualFps.toFixed(1)} < ${MIN_FPS}). Try better lighting or a faster device.`
          );
          return;
        }

        if (!decision.usable) {
          setStatus("Finished");
          setResult(`Video quality check failed: ${decision.reason}`);
          // Still send to backend for logging even if not usable
        }

        const payload = {
          fps: parseFloat(actualFps.toFixed(2)), // e.g. 13.4
          sequence: sequence,
        };
        
        const token = localStorage.getItem("token");

        fetch("http://localhost:8000/api/v1/infer/video", {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            ...(token ? { "Authorization": `Bearer ${token}` } : {})
          },
          body: JSON.stringify(payload),
        })
          .then((res) => res.json())
          .then((data) => {
            console.log("Backend response:", data);
            setResult((prev) =>
              prev + ` | Score: ${data.score ?? data.detail ?? "N/A"}`
            );
          })
          .catch((err) => {
            console.error("POST failed:", err);
          });

        setStatus(`Finished (${actualFps.toFixed(1)} FPS, ${sequence.length} frames)`);
        if (decision.usable) {
          setResult(`Usable: true | Reason: OK`);
        }

        return;
      }

      animationId = requestAnimationFrame(loop);
    }

    setup();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1>Edge Preprocessing Recorder</h1>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        width={480}
      />

      <canvas
        ref={canvasRef}
        style={{ display: "none" }}
      />

      <p>Status: {status}</p>
      <p>{result}</p>
    </div>
  );
}

