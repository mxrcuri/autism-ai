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
  pose: any | null;
  head: any | null;
  gaze: any | null;
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
    const TARGET_FPS = 15;
    const FRAME_INTERVAL = 1000 / TARGET_FPS;

    let lastFrameTime = 0;
    let startTime = 0;

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

        // ---- STEP 2: BODY POSE ----
        sendPoseFrame(videoRef.current);
        const pose = isValid ? getLatestPose() : null;

        // ---- STEP 3: FACE LANDMARKS ----
        sendFaceLandmarkFrame(videoRef.current);
        const faceMesh = getLatestFaceLandmarks();

        // ---- STEP 4: HEAD POSE ----
        const head = isValid
          ? extractHeadPose(faceMesh, isValid)
          : null;

        // ---- STEP 5: GAZE ----
        const gaze = head ? estimateGaze(head) : null;

        // ---- STORE FRAME ----
        sequence.push({
          valid: isValid,
          pose: isValid ? pose : null,
          head: isValid ? head : null,
          gaze: isValid ? gaze : null,
        });
      }

      if (now - startTime >= RECORDING_DURATION_MS) {
        const decision = evaluateVideoQuality(
          validMask,
          stats,
          50,
          0.7,
          3.0,
          TARGET_FPS
        );

        const actualFps =
          sequence.length / (RECORDING_DURATION_MS / 1000);

        const payload = {
          fps: actualFps,
          sequence: sequence,
        };
        fetch("http://localhost:8000/infer", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        })
          .then((res) => res.json())
          .then((data) => {
            console.log("Backend response:", data);
          })
          .catch((err) => {
            console.error("POST failed:", err);
          });
        

        setStatus("Finished");
        setResult(
          `Usable: ${decision.usable} | Reason: ${decision.reason}`
        );

        console.log("FINAL PAYLOAD:", payload);
        console.log("Total frames:", sequence.length);
        console.log("Actual FPS:", actualFps);


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
